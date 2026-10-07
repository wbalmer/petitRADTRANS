"""Stores the Radtrans object."""

import copy
import functools
import inspect
import os
import warnings
from typing import Any, Callable

import jax.numpy as jnp
from jax import core as jax_core
from jax import jit, vmap, lax, tree_util, debug
from jax.typing import ArrayLike

import numpy as np
import numpy.typing as npt

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.radtrans_core.interpolate_line_opacities import (
    interpolate_species_opacities,
    interpolate_species_opacities_summed,
)
from petitRADTRANS.radtrans_core import radiative_transfer_core
from petitRADTRANS.radtrans_core import linalg
from petitRADTRANS.radtrans_core.spectrum_utils import compute_bins_edges
from petitRADTRANS.opacities.opacities import (
    CIAOpacity,
    CloudOpacity,
    LineByLineOpacity,
    Opacity,
)
from petitRADTRANS.opacities import cia, clouds, continuum, hminus, load, rosseland
from petitRADTRANS.opacities.optical_depths import compute_optical_depths
from petitRADTRANS.physics import (
    flux_hz2flux_cm,
    frequency2wavelength,
    hz2um,
    um2hz,
    wavelength2frequency,
)
from petitRADTRANS.utils import LockedDict, tuple_init
from petitRADTRANS.validation import (
    _check_anisotropic_cloud_scattering,
    _check_line_opacity_mode,
    _check_path_input_data,
    _check_pressures,
    _check_clouds_particle_porosity_factors,
    _check_power_law_opacity_completeness,
    _check_target_particle_radii_completeness,
    _sanitize_emission_surface_inputs,
    _sanitize_flux_opacity_inputs,
    _check_scattering_in_emission_dtype,
    _warn_scattering_in_emission_without_scattering_species,
    _check_wavelength_boundaries,
    _handle_grid_misalignment,
    _warn_partial_cloud_coverage_without_patchy_clouds,
    _warn_opaque_cloud_top_pressure_with_scattering
)

# pRT does its computation at single precision, where possible,
# achieving a higher speed and smaller memory footprint
# (WORKING_JAX_DTYPE). FEAUTRIER_JAX_DTYPE is jnp.float64
# if config.jax_enable_x64 is True. In general JAX_DTYPE = FEAUTRIER_JAX_DTYPE
from petitRADTRANS.jax_dtype import (
    FEAUTRIER_JAX_DTYPE,
    WORKING_JAX_DTYPE,
)

# JAX functions below may have "sanitized inputs", ensuring that
# the inputs into traced functions always looks the same
# (especially important for keyword arguments
# that may sometimes be None and sometimes be something else). Empty keyword
# values use these _MISSING_X variables to fill optional keyword arguments
# with something JAX can easily handle.
_MISSING_FLOAT = jnp.array(-1.0, dtype=WORKING_JAX_DTYPE)
_MISSING_NEG_INF = jnp.array(-jnp.inf, dtype=WORKING_JAX_DTYPE)


def _convert_scalar_to_strong_type(value):
    """Normalize an optional numeric scalar to a strongly-typed jnp array.
    Some nominal values for optional keyword arguments of functions are defined as weakly typed
    Python floats, say. Depending on what kind of scalar the user gives to the function
    (Python weakly typed, strongly typed, different precision, etc.) this may cause a recompilation.
    Converting all scalars to strongly typed (length 1) JAX arrays before passing them
    to the jit compiled kernel inside a function circumvents this issue.
    None passes through unchanged (this is taken care of in the sanitizing functions
    inside the kernels).
    """

    # TODO for the future:
    #     According to Claude repeated calls of this functions in every model call (say ~10 times, and
    #     outside of jit, which is what happens in the wrapper below) may add up to 0.5 ms.
    #     So we may start feeling this in our fastest forward models on the GPU.

    if value is None:
        return None

    return jnp.asarray(value, dtype=FEAUTRIER_JAX_DTYPE)


def _handle_invalid_reference_gravity(reference_gravity):
    '''Raises an error if the reference gravity is negative.
    '''
    # Such a check raising an error can only work outside the jitted function
    # and is therefore not done in the actually jitted function kernels.
    # (debug.callbacks are an option, but these are slow.)
    if reference_gravity <= 0:
        raise ValueError(
            f"reference gravity must be > 0, but was {reference_gravity}"
        )


def _additional_opacities_placeholder_fn(wavelengths, pressures, **kwargs):
    """Returns a zero opacity matrix with shape (n_wavelengths, n_pressures).
    This is needed for the input normalization (in functions _sanitize_*)
    in the case that a user does not specify an additional_absorption_opacities_function
    or additional_scattering_opacities_function."""
    return jnp.zeros((wavelengths.size, pressures.size), dtype=WORKING_JAX_DTYPE)


def _interpolate_linear_sorted(x_grid,
                               y_grid,
                               x_query,
                               clip_xquery = False):
    """Linearly interpolate ``y(x)`` on a sorted 1D grid for to evaluate estimate y(x_query).
    If clip_xquery is True, extrapolation is not allowed and the y_grid values at either x_grid[0]
    or x_grid[-1] are returned."""
    upper_index = jnp.clip(
        jnp.searchsorted(x_grid, x_query, side="left"),
        1,
        x_grid.size - 1,
    )
    lower_index = upper_index - 1

    x0 = x_grid[lower_index]
    x1 = x_grid[upper_index]
    y0 = y_grid[lower_index]
    y1 = y_grid[upper_index]

    slope = jnp.where(x1 == x0, 0.0, (y1 - y0) / (x1 - x0))

    x_use = x_query
    if clip_xquery:
        x_use = jnp.clip(
            x_query,
            x_grid[0],
            x_grid[-1]
        )

    return y0 + slope * (x_use - x0)


# Below we define the static arguments of the pRT calculate_flux
# and calculate_transit_radii functions. They select how a function
# is traced or they drive a host-side lookup that needs a concrete value:
# the stellar parameters below reach phoenix.compute_star_spectrum,
# which indexes the PHOENIX table with star_effective_temperature.
# TODO: stellar parameters should not be traced, since we may want to
#   retrieve on them! This should be straightforward to do by reading
#   the stellar flux tables analogous to how opacities are read before
#   computations start.
# Everything *not* listed is traced, which is what makes the physical inputs
# (temperatures, mass fractions, gravity, radii, the whole cloud parameter group,
# ...) differentiable through the public functions.
_SPECTRAL_STATIC_ARGNAMES_COMMON = (
    "cloud_particle_radius_distribution",
    "patchy_clouds",
    "additional_absorption_opacities_function",
    "additional_scattering_opacities_function",
    "frequencies_to_wavelengths",
    "validate_inputs",
    "return_contribution",
    "return_clear_spectrum",
    "return_cloud_contribution",
    "return_opacities",
    "return_abundances",
    "return_particle_radii_bins",
    "return_particle_radii",
    "make_analytical_clouds_patchy",
)

_FLUX_STATIC_ARGNAMES = _SPECTRAL_STATIC_ARGNAMES_COMMON + (
    "irradiation_geometry",
    "star_effective_temperature",
    "star_radius",
    "orbit_semi_major_axis",
    "adaptive_feautrier_iterations",
    "chunk_feautrier",
    "feautrier_chunk_size",
    "thomas_min_batch",
    "ck_chunk_size",
    "return_photosphere_radius",
    "return_stellar_intensities",
    "return_angle_dependent_intensities",
)

_TRANSIT_RADII_STATIC_ARGNAMES = _SPECTRAL_STATIC_ARGNAMES_COMMON + (
    "variable_gravity",
    "return_radius_hydrostatic_equilibrium",
    "return_particle_densities",
)

_CLOUD_SPECIES_SEQUENCE_ARGNAMES = ("patchy_clouds",)

# Traced scalar arguments, normalized to strongly-typed JAX arrays *before* the
# compiled call. This must happen outside the jit boundary: otherwise a Python
# float, a numpy float64 and a jnp float32 spelling of the same value produce
# different avals and each triggers its own compilation.
_SPECTRAL_SCALAR_ARGNAMES = (
    "reference_gravity",
    "reference_pressure",
    "planet_radius",
    "opaque_cloud_top_pressure",
    "haze_factor",
    "power_law_opacity_350nm",
    "power_law_opacity_coefficient",
    "gray_opacity",
    "cloud_photosphere_median_optical_depth",
    "cloud_fraction",
    "star_irradiation_angle",
)


def _normalize_cloud_species_sequence(value):
    """Make a cloud-species selection hashable so it can be a static argument."""
    if value is None or isinstance(value, tuple):
        return value

    if isinstance(value, (str, bytes)):
        return (value,)

    return tuple(value)


def _compile_spectral_function(static_argnames):
    """Decorator factory that returns a decorator function which
    can then be immediately used on the decorated function (see
    how calculate_flux and calculate_transit_radii are decorated).
    This is done instead of @jit(static_argnames) to enable cheap
    non-jitted checks on user inputs to a function which depend on
    input variable *values*, and not their shape,
    such that we could not check without a costly debug.callback(), otherwise.
    So jit() is still being used, but with a thin so-called wrapper that
    performs the aforementioned checks.
    Compared to what we did before (only kernel functions within the now decorated
    functions we jitted), the wrapped function becomes a single compiled call.
    This can be more efficient because of how JAX / XLA compilation works.
    It also enables the computation of gradients.

    The following checks and operations are run in the wrapper:
    - the cloud porosity range check needs, which needs concrete floats (it skips tracers);
    - the power-law completeness check, which needs to see ``None``;
    - the partial-cloud-coverage warning converts ``cloud_fraction`` with
      ``float()``, which fails on a tracer;
    - the reference-gravity check must raise a plain ``ValueError``. Left to the
      compiled body it would only be reachable through a ``debug.callback``,
      which wraps it in a ``JaxRuntimeError`` and costs a host-device round-trip on
      every evaluation.

    The shell also normalizes arguments whose *representation* would otherwise
    split the compilation cache (scalar dtypes, cloud-species sequences).
    """

    def decorate(method):

        # The signature allows the wrapper below to link the arguments to their names
        signature = inspect.signature(method)

        # This is the actual compiled function.
        compiled_method = jit(method, static_argnames=static_argnames)

        # The functools.wraps(method) decorator, apply to wrapper, allows the
        # wrapper() function to impersonate the passed method function by copying
        # over all function "meta data".
        # So, if method was calculate_flux, wrapper.__name__ will return calculate_flux.
        # The fact that wrapper takes in (*args, **kwargs) means that we can hand all
        # possible arguments and keyword arguments that, for example, calculate_flux()
        # would take, over to wrapper, too.
        @functools.wraps(method)
        def wrapper(*args, **kwargs):

            # returns a "bound arguments" object, which will associate
            # the arguments of the wrapper call to the arguments that
            # method expects.
            # Example:
            '''
            def f(x,y,z,yo=None):
                return
            test = inspect.signature(f)
            test.bind(1,2,3,yo='Deine Muddah')
                Out[24]: <BoundArguments (x=1, y=2, z=3, yo='Deine Muddah')>
            '''
            bound_arguments = signature.bind(*args, **kwargs)
            # Apply default values of method if they are not given as kwargs.
            # Example, using the function above:
            '''
            hehe = test.bind(1,2,3)
            hehe
                Out[29]: <BoundArguments (x=1, y=2, z=3)>
            hehe.apply_defaults()
            hehe
                Out[31]: <BoundArguments (x=1, y=2, z=3, yo=None)>'''
            bound_arguments.apply_defaults()
            # Returns function arguments as dictionary
            # Example with same function as above
            '''
            arguments = hehe.arguments
            arguments
                Out[33]: {'x': 1, 'y': 2, 'z': 3, 'yo': None}
            isinstance(arguments, dict)
                Out[34]: True
            '''
            arguments = bound_arguments.arguments
            # Remove self as argument from the dictionary, store in self
            self = arguments.pop("self")

            # Look for patchy_clouds:
            # These must be converted into tuples, which is required
            # for having them static arguments (lists are not hashable!)
            # Technically, any hashable but mutable container works
            # for static_argument arguments that are sequences,
            # but immutability safeguards against compiler cache issues.
            # Tuples give us both: hashable and immutable
            for argname in _CLOUD_SPECIES_SEQUENCE_ARGNAMES:
                if argname in arguments:
                    arguments[argname] = _normalize_cloud_species_sequence(
                        arguments[argname]
                    )

            # Check whether porosity values make sense
            _check_clouds_particle_porosity_factors(
                arguments.get("clouds_particle_porosity_factor")
            )
            # Check wether power law opacity is correctly used
            _check_power_law_opacity_completeness(
                arguments.get("power_law_opacity_350nm"),
                arguments.get("power_law_opacity_coefficient"),
            )
            _check_target_particle_radii_completeness(
                arguments.get("target_particle_radii"),
                arguments.get("target_pressure"),
            )

            _warn_partial_cloud_coverage_without_patchy_clouds(
                cloud_fraction=arguments.get("cloud_fraction", 1.0),
                patchy_clouds=self._normalize_patchy_clouds(
                    patchy_clouds=arguments.get("patchy_clouds"),
                ),
                has_partial_coverage_analytical_cloud=(
                    self._has_partial_coverage_analytical_cloud(
                        opaque_cloud_top_pressure=arguments.get(
                            "opaque_cloud_top_pressure"
                        ),
                        power_law_opacity_350nm=arguments.get(
                            "power_law_opacity_350nm"
                        ),
                        power_law_opacity_coefficient=arguments.get(
                            "power_law_opacity_coefficient"
                        ),
                        make_analytical_clouds_patchy=arguments.get(
                            "make_analytical_clouds_patchy"
                        ),
                    )
                ),
            )

            # Warn user if they use the opaque cloud top pressure together
            # in scattering in emission == True mode.
            _warn_opaque_cloud_top_pressure_with_scattering(
                opaque_cloud_top_pressure=arguments.get("opaque_cloud_top_pressure"),
                scattering_in_emission=self._scattering_in_emission,
            )

            # Validate the reference gravity here while it is still concrete, so
            # the caller gets a plain ValueError. Only a gravity that the caller
            # is itself tracing (due to calculating a gradient with respect
            # to gravity) will not be checked.
            reference_gravity = arguments.get("reference_gravity")

            if arguments.get("validate_inputs", True) and not isinstance(
                reference_gravity, jax_core.Tracer
            ):
                _handle_invalid_reference_gravity(float(reference_gravity))

            # Validate user-supplied pressures here while still concrete (same
            # reasoning as for reference_gravity above). A None value means the
            # pressures the Radtrans object was initialized with are to be used
            # instead, which have already been checked in __init__/the pressures
            # setter, so there is nothing to do here in that case.
            pressures = arguments.get("pressures")

            if (
                arguments.get("validate_inputs", True)
                and pressures is not None
                and not isinstance(pressures, jax_core.Tracer)
            ):
                _check_pressures(pressures)

            # Must happen outside the compiled call: see _SPECTRAL_SCALAR_ARGNAMES,
            # otherwise we won't benefit from it
            for argname in _SPECTRAL_SCALAR_ARGNAMES:
                if argname in arguments:
                    arguments[argname] = _convert_scalar_to_strong_type(
                        arguments[argname]
                    )

            # None and {} both mean "no extra arguments for the user-supplied opacity
            # functions", but they are different pytrees and would therefore compile
            # separately. Normalize so that both share one compilation cache entry.
            # The {} default of get() leaves methods without that argument untouched.
            if arguments.get("additional_opacity_functions_kwargs", {}) is None:
                arguments["additional_opacity_functions_kwargs"] = {}

            # After checks have been run, rebuild the compiled method again,
            # including self.
            return compiled_method(self, **arguments)

        # Expose the compiled method so callers (and tests) can inspect the compilation, e.g.
        # Radtrans.calculate_flux.compiled_function._cache_size() to check that
        # repeated calls are not recompiling. Otherwise these quantities would
        # not acessible from the wrapper that uses the compiled function, simply.
        wrapper.compiled_function = compiled_method
        wrapper.static_argnames = tuple(static_argnames)

        return wrapper

    return decorate


class Radtrans:
    r"""Calculate spectra using a given set of opacities.

    Args:
        pressures (Optional):
            (bar) Array defining the pressure grid to be used for the spectral calculations.
        wavelength_boundaries (Optional):
            List containing left and right border of wavelength region to be considered, in micron. If nothing else
            is specified, it will be equal to ``[0.05, 300]``, hence using the full petitRADTRANS wavelength range
            (0.11 to 250 microns for ``'c-k'`` mode, 0.3 to 30 microns for the ``'lbl'`` mode). The larger the
            range the longer the computation time.
        line_species (Optional):
            List of strings, denoting which line absorber species to include.
        gas_continuum_contributors (Optional):
            List of strings, denoting which continuum absorber species to include.
        rayleigh_species (Optional):
            List of strings, denoting which Rayleigh scattering species to include.
        cloud_species (Optional):
            List of strings, denoting which cloud opacity species to include.
        line_opacity_mode (Optional[string]):
            If equal to ``'c-k'``: use low-resolution mode, at :math:`\\lambda/\\Delta \\lambda = 1000`, with the
            correlated-k assumption. if equal to ``'lbl'``: use high-resolution mode, at
            :math:`\\lambda/\\Delta \\lambda = 10^6`, with a line-by-line treatment.
        line_by_line_opacity_sampling (Optional[int]):
            If ``line_opacity_mode = 'lbl'``, then this will only load every line_by_line_opacity_sampling-nth point of the
            high-resolution opacities. This can be used to save computation time and RAM.
            The user should verify whether this leads to spectra which are sufficiently close to the results of the
            calculation at full :math:`\\lambda/\\Delta \\lambda = 10^6` resolution when binned down.
        scattering_in_emission (Optional[bool]):
            Will be ``False`` by default.
            If ``True`` scattering will be included in the emission spectral calculations. Note that this increases
            the runtime of pRT!
            Requires 64-bit JAX: the Feautrier scattering solver diverges in
            single precision, so ``True`` raises ``ValueError`` unless
            ``jax.config.update("jax_enable_x64", True)`` was called before
            petitRADTRANS was imported.
        emission_angle_grid (Optional):
            Array defining the cosines of the angle grid to be used for the emission spectrum calculations,
            and their quadrature weights for angle integrations. This is the cosine of the angle between a
            ray of light and the atmospheric normal vector, which can have values between 0 and 1.
            The array is of shape (2, n_angles), with emission_angle_grid[0] being the cosines of
            the angles, and emission_angle_grid[1] being the quadrature weights. By default pRT
            uses 3-point Gaussian quadrature grid. A dictionary of array with keys 'cos_angles' and 'weights'
            can also be used. If None, a default set of values and weights are used.
        anisotropic_cloud_scattering (Optional[bool, str]):
            If True, the anisotropic cloud scattering approximation is used for the spectral calculations.
            This is done by multiplying the scattering opacity by (1-g), where g is the scattering anisotropy.
            If False, isotropic cloud scattering opacities are used.
            If 'auto' (recommended), anisotropic_cloud_scattering is set to True for emission spectrum
            calculations, and to False for transmission spectrum calculations.
        retain_unstacked_line_opacities_for_plotting (Optional[bool]):
            To simplify the implementation of the opacity handling (P-T interpolation)
            in the spectral calculations kernels, all opacities in pRT (even if they have different
            P-T grids) are combined into one big matrix of fixed size per species.
            For species with smaller grids, unnecessary matrix values are set to 0.
            This avoids looping over species in the jitted version of the code.
            The danger is that one finely resolved (in P-T) species can balloon memory usage.
            TODO Paul: check if implementation of interpolation is done correctly.
            If retain_unstacked_line_opacities_for_plotting is True,
            keep per-species unstacked line opacity tables after the
            stacked runtime layout has been built. This is intended for plotting
            and debugging only and increases memory usage.
        path_input_data (Optional[str]):
            Path to the input_data folder, containing the files to be loaded by petitRADTRANS.
    """

    __line_opacity_property_setting_warning_message = (
        "setting a Radtrans line opacity property should be avoided\n"
        "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
        "(they need to be updated for consistency)\n"
        "It is recommended to create a new Radtrans instance instead"
    )

    __property_setting_warning_message = (
        "setting a Radtrans property directly is not recommended\n"
        "Create a new Radtrans instance (recommended) "
        "or re-do all the setup steps necessary for the modification to be taken into account"
    )

    _frequency_grid_misalignment_tolerance: float = 2e-3

    def __init__(
        self,
        pressures: ArrayLike = None,
        wavelength_boundaries: ArrayLike = None,
        line_species: list[str] | tuple[str] = None,
        gas_continuum_contributors: list[str] | tuple[str] = None,
        rayleigh_species: list[str] | tuple[str] = None,
        cloud_species: list[str] | tuple[str] = None,
        line_opacity_mode: str = "c-k",
        line_by_line_opacity_sampling: int = 1,
        scattering_in_emission: bool = True,
        emission_angle_grid: ArrayLike = None,
        anisotropic_cloud_scattering: bool = "auto",  # NOTE: Probably doesn't need to be a main parameter
        retain_unstacked_line_opacities_for_plotting: bool = False,
        path_input_data: str = None,
    ):
        # Inputs checks
        _check_line_opacity_mode(line_opacity_mode)
        _check_anisotropic_cloud_scattering(anisotropic_cloud_scattering)

        # Initialize properties
        if pressures is None:
            warnings.warn("pressure was not set, initializing one layer at 1 bar")
            pressures = jnp.array([1.0], dtype=FEAUTRIER_JAX_DTYPE)  # bar

        self._pressures = (
            jnp.array(pressures, dtype=FEAUTRIER_JAX_DTYPE) * 1e6
        )  # bar to cgs  # TODO pressure could be spectral function argument
        _check_pressures(pressures)

        self._line_species = tuple_init(line_species)
        self._gas_continuum_contributors = tuple_init(gas_continuum_contributors)
        self._rayleigh_species = tuple_init(rayleigh_species)
        self._cloud_species = tuple_init(cloud_species)

        self._line_opacity_mode = line_opacity_mode
        self._line_by_line_opacity_sampling = line_by_line_opacity_sampling
        self._scattering_in_emission = scattering_in_emission
        self._retain_unstacked_line_opacities_for_plotting = retain_unstacked_line_opacities_for_plotting

        # Refuse single-precision emission scattering up front, before the
        # (slow) opacity load, rather than returning a diverged spectrum.
        _check_scattering_in_emission_dtype(self._scattering_in_emission)

        _warn_scattering_in_emission_without_scattering_species(
            self._scattering_in_emission,
            self._cloud_species,
            self._rayleigh_species
        )

        if wavelength_boundaries is None:
            self._wavelength_boundaries = jnp.array(
                [0.05, 300.0], dtype=WORKING_JAX_DTYPE
            )  # um
        else:
            _check_wavelength_boundaries(wavelength_boundaries)
            self._wavelength_boundaries = jnp.asarray(
                wavelength_boundaries, dtype=WORKING_JAX_DTYPE
            )

        self._anisotropic_cloud_scattering = anisotropic_cloud_scattering

        # Pre-resolve whether opacities of species can be summed in the first
        # column such that kernel methods do not read mutable instance states at call time.
        # TODO Paul: check if self._sum_opacities_emission is really needed. _set_sum_opacities() is used
        #   in other functions below, but self._sum_opacities_emission is only used once,
        #   in calculate_flux_autodiff, which Evert said should not still be there.
        self._sum_opacities_emission: bool = self._set_sum_opacities(
            emission=True,
            line_opacity_mode=self._line_opacity_mode,
            scattering_in_emission=self._scattering_in_emission,
        )
        self._anisotropic_cloud_scattering_emission: bool = (
            self._anisotropic_cloud_scattering in (True, "auto")
        )

        if path_input_data is None:
            path_input_data = petitradtrans_config_parser.get_input_data_path()
        self._path_input_data = path_input_data

        # Initialize non-settable properties
        self.__sum_opacities = False
        self.__absorber_present = not (
            len(self._line_species) == 0
            and len(self._gas_continuum_contributors) == 0
            and len(self._rayleigh_species) == 0
            and len(self._cloud_species) == 0
        )

        # Initialize line parameters
        # TODO: only c-k species allow user to request a wavelength grid that is larger
        #   than the opacity file range. For line-by-line, CIA and cloud it crashes.
        #   Look into a fix!
        self._frequencies, self._frequency_bins_edges = self._init_frequency_grid()
        self._wavelengths_micron = jnp.array(cst.c / self._frequencies / 1e-4, dtype=WORKING_JAX_DTYPE)

        # Initialize loaded line opacities variables
        self._lines_loaded_opacities = LockedDict.build_and_lock(
            {
                "temperature_pressure_grid": {},
                "temperature_grid_size": {},
                "pressure_grid_size": {},
                "opacity_grid": {},
                "temperature_pressure_grid_stacked": jnp.zeros(
                    (0, 1, 1, 2), dtype=WORKING_JAX_DTYPE
                ),
                "temperature_grid_size_array": jnp.zeros(0, dtype=jnp.int32),
                "pressure_grid_size_array": jnp.zeros(0, dtype=jnp.int32),
                "opacity_grid_stacked": jnp.zeros(
                    (0, 1, 1, 1), dtype=WORKING_JAX_DTYPE
                ),
                "ck_g_combination_weights": jnp.ones(1, dtype=WORKING_JAX_DTYPE),
                "ck_g_combination_weights_sum": jnp.array(
                    1.0, dtype=WORKING_JAX_DTYPE
                ),
                "g_gauss": jnp.ones(1, dtype=WORKING_JAX_DTYPE),
                "weights_gauss": jnp.ones(1, dtype=WORKING_JAX_DTYPE),
            }
        )

        # Initialize cia opacities
        self._cias_loaded_opacities = cia.init_cia_loaded_opacities(
            cia_contributors=cia.get_cia_contributors(self._gas_continuum_contributors)
        )

        # Initialize loaded cloud opacities variables
        # Use JAX-friendly empty sentinel arrays as defaults so jitted code sees arrays, not None
        empty_wavelength_shape = 3
        empty_n_clouds_shape = 1
        empty_n_particle_radii = 3
        empty_3d = jnp.zeros(
            (empty_n_particle_radii, empty_wavelength_shape, empty_n_clouds_shape),
            dtype=WORKING_JAX_DTYPE,
        )
        self._clouds_loaded_opacities = LockedDict.build_and_lock(
            {
                "wavelengths": jnp.zeros(
                    empty_wavelength_shape, dtype=WORKING_JAX_DTYPE
                ),
                "absorption_opacities": empty_3d,
                "scattering_opacities": empty_3d,
                "particle_radii": jnp.zeros(
                    empty_n_particle_radii, dtype=WORKING_JAX_DTYPE
                ),
                "particle_radii_bins": jnp.zeros(
                    empty_n_particle_radii + 1, dtype=WORKING_JAX_DTYPE
                ),
                "particle_densities": jnp.zeros(
                    empty_n_clouds_shape, dtype=WORKING_JAX_DTYPE
                ),
                "particle_asymmetry_parameters": empty_3d,
            }
        )

        # Initialize the angle (mu) grid for the emission spectral calculations
        self._emission_angle_grid = LockedDict.build_and_lock(
            {"cos_angles": None, "weights": None}
        )

        if emission_angle_grid is None:
            # 3-point Gaussian quadrature grid for mu, which is ranging from 0 to 1.
            self._emission_angle_grid["cos_angles"] = jnp.array(
                [0.1127016654, 0.5, 0.8872983346],
                dtype=WORKING_JAX_DTYPE,
            )
            # Corresponding weights of the same grid.
            self._emission_angle_grid["weights"] = jnp.array(
                [0.2777777778, 0.4444444444, 0.2777777778],
                dtype=WORKING_JAX_DTYPE,
            )
        else:
            if isinstance(emission_angle_grid, dict):
                self._emission_angle_grid["cos_angles"] = jnp.asarray(
                    emission_angle_grid["cos_angles"], dtype=WORKING_JAX_DTYPE
                )
                self._emission_angle_grid["weights"] = jnp.asarray(
                    emission_angle_grid["weights"], dtype=WORKING_JAX_DTYPE
                )
            elif hasattr(emission_angle_grid, "__iter__"):
                self._emission_angle_grid["cos_angles"] = jnp.array(
                    emission_angle_grid[0], copy=True, dtype=WORKING_JAX_DTYPE
                )
                self._emission_angle_grid["weights"] = jnp.array(
                    emission_angle_grid[1], copy=True, dtype=WORKING_JAX_DTYPE
                )

                if (
                    self._emission_angle_grid["cos_angles"].size
                    != self._emission_angle_grid["weights"].size
                ):
                    raise ValueError(
                        f"emission_cos_angle_grid_weights must be an array of the same size than "
                        f"emission_cos_angle_grid ({self._emission_angle_grid['cos_angles'].size}), "
                        f"but is of size {self._emission_angle_grid['weights'].size}"
                    )
            else:
                raise ValueError(
                    f"emission_angle_grid must be a dictionary or an iterable of shape (2, n_angles), but "
                    f"is of type {type(emission_angle_grid)}"
                )

        # Load all opacities
        self.load_all_opacities()

    @property
    def anisotropic_cloud_scattering(self):
        return self._anisotropic_cloud_scattering

    @anisotropic_cloud_scattering.setter
    def anisotropic_cloud_scattering(self, mode: str):
        warnings.warn(self.__property_setting_warning_message)
        _check_anisotropic_cloud_scattering(mode)
        self._anisotropic_cloud_scattering = mode

    @property
    def cias_loaded_opacities(self):
        return self._cias_loaded_opacities

    @cias_loaded_opacities.setter
    def cias_loaded_opacities(self, dictionary: dict[str, dict[str, ArrayLike]]):
        warnings.warn(
            "setting the Radtrans CIA opacity manuallyshould be avoided\n"
            "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
            "(they need to be updated for consistency)\n"
            "It is recommended to create a new Radtrans instance instead"
        )
        for key1, value1 in dictionary.items():
            for key2, value2 in value1.items():
                self._cias_loaded_opacities[key1][key2] = value2

    @property
    def clouds_loaded_opacities(self):
        return self._clouds_loaded_opacities

    @clouds_loaded_opacities.setter
    def clouds_loaded_opacities(self, dictionary: dict[str, ArrayLike]):
        warnings.warn(
            "setting the Radtrans cloud opacity property should be avoided\n"
            "These properties are loaded from the opacity data in the input_data directory and are inter-dependent "
            "(they need to be updated for consistency)\n"
            "It is recommended to create a new Radtrans instance instead"
        )
        for key, value in dictionary.items():
            self._clouds_loaded_opacities[key] = value

    @property
    def cloud_species(self):
        return self._cloud_species

    @cloud_species.setter
    def cloud_species(self, species: list):
        warnings.warn(self.__property_setting_warning_message)
        _warn_scattering_in_emission_without_scattering_species(
            scattering_in_emission=value,
            cloud_species=self._cloud_species,
            rayleigh_species=self._rayleigh_species
        )
        self._cloud_species = species

    @property
    def emission_angle_grid(self):
        return self._emission_angle_grid

    @emission_angle_grid.setter
    def emission_angle_grid(self, dictionary: dict[str, ArrayLike]):
        warnings.warn(self.__property_setting_warning_message)

        for key, value in dictionary.items():
            self._emission_angle_grid[key] = value

    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.setter
    def frequencies(self, array: ArrayLike):
        warnings.warn(
            "setting frequencies directly should be avoided\n"
            "This property is loaded from the opacity data in the input_data directory "
            "and is inter-dependent with other line opacities parameters, that also need to be updated\n"
            "It is recommended to create a new Radtrans instance with e.g. different wavelength boundaries instead"
        )
        self._frequencies = array

    @property
    def frequency_bins_edges(self):
        return self._frequency_bins_edges

    @frequency_bins_edges.setter
    def frequency_bins_edges(self, array: ArrayLike):
        warnings.warn(self.__line_opacity_property_setting_warning_message)
        self._frequency_bins_edges = array

    @property
    def gas_continuum_contributors(self):
        return self._gas_continuum_contributors

    @gas_continuum_contributors.setter
    def gas_continuum_contributors(self, species: list[str]):
        warnings.warn(self.__property_setting_warning_message)
        self._gas_continuum_contributors = species

    @property
    def line_by_line_opacity_sampling(self):
        return self._line_by_line_opacity_sampling

    @line_by_line_opacity_sampling.setter
    def line_by_line_opacity_sampling(self, value: int):
        warnings.warn(self.__property_setting_warning_message)
        self._line_by_line_opacity_sampling = value

    @property
    def lines_loaded_opacities(self):
        return self._lines_loaded_opacities

    @lines_loaded_opacities.setter
    def lines_loaded_opacities(self, dictionary: dict[str, ArrayLike]):
        warnings.warn(self.__line_opacity_property_setting_warning_message)

        for key, value in dictionary.items():
            self._lines_loaded_opacities[key] = value

    @property
    def line_opacity_mode(self):
        return self._line_opacity_mode

    @line_opacity_mode.setter
    def line_opacity_mode(self, mode):
        warnings.warn(self.__property_setting_warning_message)
        _check_line_opacity_mode(mode)
        self._line_opacity_mode = mode

    @property
    def line_species(self):
        return tuple(self._line_species)

    @line_species.setter
    def line_species(self, species: list | tuple):
        warnings.warn(self.__property_setting_warning_message)
        self._line_species = tuple(species)

    @property
    def path_input_data(self):
        return self._path_input_data

    @path_input_data.setter
    def path_input_data(self, path: str):
        warnings.warn(self.__property_setting_warning_message)
        _check_path_input_data(path)
        self._path_input_data = path

    @property
    def pressures(self) -> ArrayLike:
        return self._pressures

    @pressures.setter
    def pressures(self, array):
        warnings.warn(self.__property_setting_warning_message)
        _check_pressures(array)
        self._pressures = array

    @property
    def rayleigh_species(self):
        return self._rayleigh_species

    @rayleigh_species.setter
    def rayleigh_species(self, species: list):
        warnings.warn(self.__property_setting_warning_message)
        _warn_scattering_in_emission_without_scattering_species(
            scattering_in_emission=value,
            cloud_species=self._cloud_species,
            rayleigh_species=self._rayleigh_species
        )
        self._rayleigh_species = species

    @property
    def scattering_in_emission(self):
        return self._scattering_in_emission

    @scattering_in_emission.setter
    def scattering_in_emission(self, value: bool):
        warnings.warn(self.__property_setting_warning_message)
        _check_scattering_in_emission_dtype(value)
        _warn_scattering_in_emission_without_scattering_species(
            scattering_in_emission=value,
            cloud_species=self._cloud_species,
            rayleigh_species=self._rayleigh_species
        )
        self._scattering_in_emission = value

    @property
    def wavelength_boundaries(self):
        return self._wavelength_boundaries

    @wavelength_boundaries.setter
    def wavelength_boundaries(self, array: ArrayLike):
        warnings.warn(self.__property_setting_warning_message)
        _check_wavelength_boundaries(array)
        self._wavelength_boundaries = array

    @staticmethod
    def _has_partial_coverage_analytical_cloud(
        opaque_cloud_top_pressure=None,
        power_law_opacity_350nm=None,
        power_law_opacity_coefficient=None,
        make_analytical_clouds_patchy=False,
    ):
        """Check whether analytical cloud options are requested and
         do therefore participate in cloud_fraction mixing (since
         this is pRT's standard treatment)."""

        return (opaque_cloud_top_pressure is not None or (
            power_law_opacity_350nm is not None
            and power_law_opacity_coefficient is not None
        )) and make_analytical_clouds_patchy

    def _normalize_patchy_clouds(
        self, patchy_clouds=None
    ):
        """Make sure that patchy_clouds is always normalized to be a tuple,
        to be consumed by the jitted radiative transfer kernels.
        """

        if patchy_clouds is None:
            return tuple()

        if isinstance(patchy_clouds, (str, bytes)):
            patchy_clouds = (patchy_clouds,)
        else:
            patchy_clouds = tuple(patchy_clouds)

        return tuple(
            species for species in self._cloud_species if species in patchy_clouds
        )

    def _get_complete_coverage_cloud_species(self, patchy_clouds):
        """Return the tuple of non-patchy cloud species.
        """

        if patchy_clouds is None:
            return tuple()

        # convert the tuple to a set because the
        # if species not in patchy_cloud_species
        # is then much cheaper, somehow (it is O(1), not O(N)).
        patchy_cloud_species = set(patchy_clouds)

        return tuple(
            species
            for species in self._cloud_species
            if species not in patchy_cloud_species
        )

    def _get_partial_coverage_cloud_species(self, patchy_clouds):
        """Return the tuple of patchy cloud species.
        """
        if patchy_clouds is None:
            return tuple() # Unreachable since patchy_clouds will have been changed to an empty tuple
                           # elsewhere, if it was None.

        return tuple(
            species for species in self._cloud_species if species in patchy_clouds
        )


    def _init_spectral_function(
        self,
        is_emission,
        patchy_clouds=None,
    ):
        """Initializations common to calculate_flux and calculate_transit_radii."""

        # Here patchy_clouds are turned into a tuple, regardless
        # of the user input.
        patchy_clouds = self._normalize_patchy_clouds(
            patchy_clouds=patchy_clouds
        )

        # Determine whether opacities will / should be mixed into one "super species".
        # This happening for scattering in emission in c-k and always for lbl mode.
        sum_opacities = self._set_sum_opacities(
            emission=is_emission,
            line_opacity_mode=self._line_opacity_mode,
            scattering_in_emission=self._scattering_in_emission,
        )

        # If auto and emission: weigh cloud opacities by (1-g) to correctly treat
        # forward scattering in the diffusive limit.
        if self._anisotropic_cloud_scattering == "auto":
            anisotropic_cloud_scattering = is_emission
        # Or turn it on otherwise if requested by the user.
        else:
            anisotropic_cloud_scattering = self._anisotropic_cloud_scattering

        return (
            anisotropic_cloud_scattering,
            patchy_clouds,
            sum_opacities,
        )

    def _stack_loaded_line_opacities(
        self,
        temperature_pressure_grids,
        temperature_grid_sizes_map,
        pressure_grid_sizes_map,
        opacity_grids,
    ):
        '''Takes in the read line opacities, each on their own T-P grid,
        and enters them all into one big matrix of shape
        (n_line_species, n_g, self._frequencies.size, max_temperature_pressure_grid_size)
        which is easier for jax functions to consume in the following.
        The max_temperature_pressure_grid_size is defined by the species with the largest
        T, P grids (T and P largest grids can come from different species).
        Interestingly: since .reshape()ing opacities into a 2-d T-P grid (after they were serial
        before) does not change the memory layout of the arrays, these operations are essentially
        free and are done below to help with readability.
        The T, P and opacity grids of all species with smaller
        P-T grids will be padded like shown in the ASCII art below.
        What is more, in the end the P-T interpolation will be done per species in pRT4,
        so there's no underlying assumption of species sharing P-T grids whatsoever.
        At least this is my current interpretation.
        # TODO paul: this is my current assumption, also after asking
        #   Claude quickly about it. Confirm this during the code review later.
        The original P-T grid sizes of all species are retained.
        So the padding works like this (note that it would be more conventional
        to have T on the y and P on the x axis, but I was too lazy to change that
        now from my first draft):
                                      T  -->
                 |-----------------------|-----------------------|
                 |          P0           |          P0           |
                 |                       |                       |
                 |  T0            Tact   |  Tact           Tact  |
                 |                       |                       |
             P   |         Pact          |         Pact          |
             |   |-----------------------|-----------------------|
             v   |         Pact          |         Pact          |
                 |                       |                       |
                 |  T0            Tact   |  Tact           Tact  |
                 |                       |                       |
                 |         Pact          |         Pact          |
                 |-----------------------|-----------------------|
        '''
        n_line_species = len(self._line_species)

        # Stores the combination of the Gauss weights for k-table mixing (normalized to add up
        # to one) and their sum before normalization.
        (
            self._lines_loaded_opacities["ck_g_combination_weights"],
            self._lines_loaded_opacities["ck_g_combination_weights_sum"],
        ) = radiative_transfer_core.build_ck_combination_state(
            self._lines_loaded_opacities["weights_gauss"]
        )

        if n_line_species == 0:
            self._lines_loaded_opacities["temperature_pressure_grid_stacked"] = (
                # For non-zero n_line_species this would be of shape
                # (n_line_species, max_temperature_grid_size, max_pressure_grid_size, 2)
                # see below.
                jnp.zeros((0, 1, 1, 2), dtype=WORKING_JAX_DTYPE)
            )
            self._lines_loaded_opacities["temperature_grid_size_array"] = jnp.zeros(
                0, dtype=jnp.int32
            )
            self._lines_loaded_opacities["pressure_grid_size_array"] = jnp.zeros(
                0, dtype=jnp.int32
            )
            self._lines_loaded_opacities["opacity_grid_stacked"] = jnp.zeros(
                # For non-zero n_line_species this would be of shape
                # (n_line_species, n_g, self._frequencies.size, max_temperature_pressure_grid_size)
                # see below.
                (0, 1, self._frequencies.size, 1), dtype=WORKING_JAX_DTYPE
            )
            return

        temperature_grid_sizes = np.array(
            [temperature_grid_sizes_map[species] for species in self._line_species],
            dtype=np.int32,
        )
        pressure_grid_sizes = np.array(
            [pressure_grid_sizes_map[species] for species in self._line_species],
            dtype=np.int32,
        )
        max_temperature_grid_size = int(np.max(temperature_grid_sizes))
        max_pressure_grid_size = int(np.max(pressure_grid_sizes))
        # Use max_T * max_P as the flat T-P dimension so that the stride used by
        # interpolate_line_opacities (= temperature_pressure_grid_input.shape[1] = max_P)
        # matches the stride of the stacked opacity grid.  Using max(T_i * P_i) instead
        # gave the wrong stride whenever species had different P grid sizes.
        max_temperature_pressure_grid_size = (
            max_temperature_grid_size * max_pressure_grid_size
        )
        n_g = int(self._lines_loaded_opacities["g_gauss"].size)

        temperature_pressure_grid_stacked = np.zeros(
            (n_line_species, max_temperature_grid_size, max_pressure_grid_size, 2),
            dtype=np.asarray(self._pressures).dtype,
        )
        opacity_grid_stacked = np.zeros(
            (
                n_line_species,
                n_g,
                self._frequencies.size,
                max_temperature_pressure_grid_size,
            ),
            dtype=np.asarray(self._frequencies).dtype,
        )

        for i_species, species in enumerate(self._line_species):
            temperature_grid_size = int(temperature_grid_sizes[i_species])
            pressure_grid_size = int(pressure_grid_sizes[i_species])
            temperature_pressure_grid = np.asarray(
                temperature_pressure_grids[species]
            ).reshape(temperature_grid_size, pressure_grid_size, 2)
            opacity_grid = np.asarray(opacity_grids[species])

            temperature_pressure_grid_stacked_species = (
                temperature_pressure_grid_stacked[i_species]
            )
            # Only fill those lower values of the grid that exist
            temperature_pressure_grid_stacked_species[
                :temperature_grid_size, :pressure_grid_size, :
            ] = temperature_pressure_grid
            if pressure_grid_size < max_pressure_grid_size:
                # fill values such that the highest P value is
                # simply repeated in the grid, for all valid
                # temperatures
                temperature_pressure_grid_stacked_species[
                    :temperature_grid_size, pressure_grid_size:, :
                ] = temperature_pressure_grid[
                    :, pressure_grid_size - 1 : pressure_grid_size, :
                ]
            if temperature_grid_size < max_temperature_grid_size:
                # fill values such that the highest T values is
                # simply repeated, for all valid pressures, including
                # the new regime where Pmax is simply repeated
                temperature_pressure_grid_stacked_species[temperature_grid_size:, :, :] = (
                    temperature_pressure_grid_stacked_species[
                        temperature_grid_size - 1 : temperature_grid_size, :, :
                    ]
                )

            # Reshape the flat (T_act * P_act) opacity dimension to 2D (T_act, P_act),
            # pad to (max_T, max_P).
            opacity_grid_2d = opacity_grid.reshape(
                n_g, self._frequencies.size, temperature_grid_size, pressure_grid_size
            )

            # This will make a "view", not a copy of opacity_grid_stacked[i_species]
            # the reshape does not hurt since this is how the elements are arranged
            # in memory anyway. So everything being done to opacity_grid_stacked_species
            # below will also be done to opacity_grid_stacked (they are identical, but
            # opacity_grid_stacked_species only sees a subsection of opacity_grid_stacked.
            opacity_grid_stacked_species = opacity_grid_stacked[i_species].reshape(
                n_g,
                self._frequencies.size,
                max_temperature_grid_size,
                max_pressure_grid_size,
            )
            opacity_grid_stacked_species[
                :, :, :temperature_grid_size, :pressure_grid_size
            ] = opacity_grid_2d
            # Do the same for the opacity values that was done for the
            # pressure temperature values.
            if pressure_grid_size < max_pressure_grid_size:
                opacity_grid_stacked_species[
                    :, :, :temperature_grid_size, pressure_grid_size:
                ] = opacity_grid_2d[:, :, :, pressure_grid_size - 1 : pressure_grid_size]
            if temperature_grid_size < max_temperature_grid_size:
                opacity_grid_stacked_species[:, :, temperature_grid_size:, :] = (
                    opacity_grid_stacked_species[
                        :, :, temperature_grid_size - 1 : temperature_grid_size, :
                    ]
                )

        self._lines_loaded_opacities["temperature_pressure_grid_stacked"] = jnp.array(
            temperature_pressure_grid_stacked,
            dtype=WORKING_JAX_DTYPE,
        )
        self._lines_loaded_opacities["temperature_grid_size_array"] = jnp.array(
            temperature_grid_sizes,
            dtype=jnp.int32,
        )
        self._lines_loaded_opacities["pressure_grid_size_array"] = jnp.array(
            pressure_grid_sizes,
            dtype=jnp.int32,
        )
        self._lines_loaded_opacities["opacity_grid_stacked"] = jnp.array(
            opacity_grid_stacked,
            dtype=WORKING_JAX_DTYPE,
        )

    def _set_unstacked_line_opacity_debug_data(
        self,
        temperature_pressure_grids,
        temperature_grid_sizes,
        pressure_grid_sizes,
        opacity_grids,
    ):
        '''
        Empty the dictionaries that pRT classically used to store opacities.
        Only fill them again if specifically requested for debugging / plotting.
        '''
        self._lines_loaded_opacities["temperature_pressure_grid"].clear()
        self._lines_loaded_opacities["temperature_grid_size"].clear()
        self._lines_loaded_opacities["pressure_grid_size"].clear()
        self._lines_loaded_opacities["opacity_grid"].clear()

        if not self._retain_unstacked_line_opacities_for_plotting:
            return

        self._lines_loaded_opacities["temperature_pressure_grid"].update(
            temperature_pressure_grids
        )
        self._lines_loaded_opacities["temperature_grid_size"].update(
            temperature_grid_sizes
        )
        self._lines_loaded_opacities["pressure_grid_size"].update(pressure_grid_sizes)
        self._lines_loaded_opacities["opacity_grid"].update(opacity_grids)

    @staticmethod
    def _set_sum_opacities(emission, line_opacity_mode, scattering_in_emission):
        """Determine whether opacities / optical depths (depends on line_opacity_mode)
        can be treated in a "combined into one species" way (either corr-k mixing or
        line-by-line adding).

        Args:
            emission (bool):
                Whether pRT is used to calculate an emission spectrum or not
            line_opacity_mode ([string]):
                Defines if pRT is in correlated-k ("c-k") or line-by-line ("lbl") opacity mode.
            scattering_in_emission (bool):
                Defines if pRT is considering scattering in emission (default).

        Return:
            Whether opacities will be treated in a combined mode, for all species.
        """
        sum_opacities = False  # False in c-k emission if no scattering, and always False in c-k transmission
                               #
        if line_opacity_mode == "c-k":
            if emission and scattering_in_emission:
                sum_opacities = True
        elif line_opacity_mode == "lbl":  # always True in lbl mode
            sum_opacities = True
        return sum_opacities

    def _resolve_solver_pressures(self, pressures=None):
        """pRT allows users to specify a pressure array different from the one used
        initialize the pRT object with. Within calculate_flux(), calculate_transit_radii()
        and _calculate_photosphere_radius() this function is called to check whether
        the user has specified a pressure array or whether the initialization pressure
        is to be used."""
        if pressures is None:
            return self._pressures

        return jnp.asarray(pressures, dtype=FEAUTRIER_JAX_DTYPE) * 1e6

    def _calculate_flux(
        self,
        pressures,
        temperatures,
        reference_gravity,
        opacities,
        continuum_opacities_scattering,
        irradiation_geometry,
        star_irradiation_cos_angle,
        stellar_intensity,
        reflectances,
        emissivities,
        cloud_f_sed,
        photospheric_cloud_optical_depths,
        cloud_photosphere_wavelength_boundaries,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        adaptive_feautrier_iterations,
        return_contribution=False,
        return_angle_dependent_intensities=False,
        sum_opacities=True,
        chunk_feautrier=True,
        feautrier_chunk_size=1536,
        thomas_min_batch=linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
    ):
        """Compute the emission flux for the already-prepared opacity state.

        This helper converts opacities into optical depths, chooses the
        appropriate radiative-transfer solver, and optionally evaluates the
        emission contribution function.

        Args:
            reference_gravity:
                Reference gravity in cgs units used in the optical-depth
                calculation.
            opacities:
                Total absorption opacity array prepared by
                ``_calculate_opacities``.
            continuum_opacities_scattering:
                Continuum scattering opacity array with frequency and layer
                dimensions.
            irradiation_geometry:
                Irradiation geometry passed to the emission solver.
            star_irradiation_cos_angle:
                Cosine of the stellar irradiation angle.
            stellar_intensity:
                Incident stellar intensity spectrum on the model frequency grid.
            reflectances:
                Surface reflectance spectrum.
            emissivities:
                Surface emissivity spectrum.
            cloud_f_sed:
                Cloud settling parameter forwarded to the optical-depth helper
                when photospheric cloud scaling is requested.
            photospheric_cloud_optical_depths:
                Optional target cloud optical depth used by the photospheric
                cloud parametrization.
            cloud_photosphere_wavelength_boundaries:
                Optional wavelength interval used when evaluating the
                photospheric cloud optical depth.
            cloud_anisotropic_scattering_opacities:
                Cloud anisotropic scattering opacities used when scattering in
                emission is enabled.
            cloud_absorption_opacities:
                Cloud absorption opacity contribution used by the optical-depth
                calculation.
            adaptive_feautrier_iterations:
                Whether to adapt the number of scattering iterations in the
                Feautrier solver.
            return_contribution:
                If ``True``, also return the emission contribution function.
            return_angle_dependent_intensities:
                If ``True``, return angle-dependent intensities. Non-scattering
                emission returns a zero placeholder with the correct shape.
            chunk_feautrier:
                If ``True``, evaluate the Feautrier solver in frequency chunks.
            feautrier_chunk_size:
                Number of frequency bins per chunk when ``chunk_feautrier`` is
                enabled.
            thomas_min_batch:
                Batch size above which the Thomas-factorisation Feautrier
                solver path is used over PCR on GPU.

        Returns:
            Tuple ``(flux, emission_contribution, optical_depths,
            relative_cloud_scaling_factor)`` on the internal frequency grid.
        """

        (
            optical_depths,
            photon_destruction_probabilities,
            relative_cloud_scaling_factor,
        ) = compute_optical_depths( # TODO review left for David
                                    #   remove TODO when done.
            pressures=pressures,
            reference_gravity=reference_gravity,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            sum_opacities=sum_opacities,
            scattering_in_emission=self._scattering_in_emission,
            absorber_present=self.__absorber_present,
            # Custom cloud parameters
            frequencies=self._frequencies,
            weights_gauss=self._lines_loaded_opacities["weights_gauss"],
            cloud_wavelengths=cloud_photosphere_wavelength_boundaries,
            cloud_f_sed=cloud_f_sed,
            cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities=cloud_absorption_opacities,
            photospheric_cloud_optical_depths=photospheric_cloud_optical_depths,
        )
        angle_dependent_intensities = None
        if self._scattering_in_emission:
            flux, emission_contribution, angle_dependent_intensities = (
                radiative_transfer_core.compute_feautrier_radiative_transfer( # TODO review left for David
                                                                              #     remove TODO when done.
                    frequencies_bin_edges=jnp.asarray(
                        self._frequency_bins_edges, dtype=FEAUTRIER_JAX_DTYPE
                    ),
                    temperatures=jnp.asarray(
                        temperatures, dtype=FEAUTRIER_JAX_DTYPE
                    ),
                    weights_gauss=jnp.asarray(
                        self._lines_loaded_opacities["weights_gauss"],
                        dtype=FEAUTRIER_JAX_DTYPE,
                    ),
                    emission_cos_angles=jnp.asarray(
                        self._emission_angle_grid["cos_angles"],
                        dtype=FEAUTRIER_JAX_DTYPE,
                    ),
                    emission_cos_angles_weights=jnp.asarray(
                        self._emission_angle_grid["weights"],
                        dtype=FEAUTRIER_JAX_DTYPE,
                    ),
                    optical_depths_in=jnp.asarray(
                        optical_depths[:, :, 0, :], dtype=WORKING_JAX_DTYPE
                    ),
                    photon_destruction_probabilities_in=jnp.asarray(
                        photon_destruction_probabilities,
                        dtype=WORKING_JAX_DTYPE,
                    ),
                    stellar_intensity=jnp.asarray(
                        stellar_intensity, dtype=FEAUTRIER_JAX_DTYPE
                    ),
                    star_irradiation_cos_angle=jnp.asarray(
                        star_irradiation_cos_angle, dtype=FEAUTRIER_JAX_DTYPE
                    ),
                    reflectances=jnp.asarray(
                        reflectances, dtype=FEAUTRIER_JAX_DTYPE
                    ),
                    emissivities=jnp.asarray(
                        emissivities, dtype=FEAUTRIER_JAX_DTYPE
                    ),
                    adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                    irradiation_geometry_str=irradiation_geometry,
                    return_contribution=return_contribution,
                    chunk_feautrier=chunk_feautrier,
                    feautrier_chunk_size=feautrier_chunk_size,
                    thomas_min_batch=thomas_min_batch,
                )
            )
        else:
            if self._line_opacity_mode == "lbl" and len(self._line_species) > 1:
                _optical_depths = optical_depths[:, :, :1, :]
            else:
                _optical_depths = optical_depths

            flux, emission_contribution = radiative_transfer_core.compute_ck_flux( # TODO review left for David
                                                                                   #     remove TODO when done.
                self._frequencies,
                temperatures,
                self._lines_loaded_opacities["weights_gauss"],
                self._emission_angle_grid["cos_angles"],
                self._emission_angle_grid["weights"],
                _optical_depths,
                return_contribution,
            )
            if return_angle_dependent_intensities:
                angle_dependent_intensities = jnp.zeros(
                    (
                        self._emission_angle_grid["cos_angles"].shape[0],
                        self._frequencies.shape[0],
                    ),
                    dtype=WORKING_JAX_DTYPE,
                )
        return (
            flux,
            emission_contribution,
            angle_dependent_intensities,
            optical_depths,
            relative_cloud_scaling_factor,
        )

    # TODO Assigned to Nora to review
    def _calculate_radius_hydrostatic_equilibrium(
        self,
        pressures,
        temperatures,
        mean_molar_masses,
        reference_gravity,
        planet_radius,
    ):
        """
        Calculate the atmospheric radius profile assuming hydrostatic equilibrium.

        Computes the atmospheric mass density at each pressure level from the
        ideal gas law (using the mean molecular mass and temperature at that
        level), then integrates the hydrostatic equilibrium equation to
        obtain the radius as a function of pressure. The last pressure level
         is used as the reference pressure, anchored to 'planet_radius'.

        Args:
            pressures (ArrayLike): Pressure at each atmospheric level.
            temperatures (ArrayLike): Temperature at each atmospheric level.
            mean_molar_masses (ArrayLike): Mean molecular mass at each
                atmospheric level.
            reference_gravity (float or ArrayLike): Gravitational acceleration
                used in the hydrostatic equilibrium calculation.
            planet_radius (float): Planet radius in cm, defining the radius at
                the reference pressure.

        Returns:
            ArrayLike: Radius at each atmospheric pressure level, consistent
            with hydrostatic equilibrium.
        """
        atmospheric_densities = (
            pressures * mean_molar_masses * cst.amu / cst.kB / temperatures
        )

        return radiative_transfer_core.compute_radius_hydrostatic_equilibrium(
            pressures=pressures,
            reference_gravity=reference_gravity,
            atmosphere_densities=atmospheric_densities,
            reference_pressure=pressures[-1],
            planet_radius=planet_radius,
        )

    def _get_photosphere_optical_depths(
        self,
        pressures,
        optical_depths,
        reference_gravity,
        opacities,
        continuum_opacities_scattering,
        cloud_f_sed,
        cloud_photosphere_median_optical_depth,
        cloud_photosphere_wavelength_boundaries,
        cloud_anisotropic_scattering_opacities,
        cloud_absorption_opacities,
        sum_opacities,
    ):
        # This is how it is used if _calculate_optical_depths
        # is called from within calculate_flux().
        if optical_depths is not None:
            return optical_depths

        # This branch never executes in that case but just
        # calls to create opacities in the normal way.
        optical_depths, _, _ = compute_optical_depths(
            pressures=pressures,
            reference_gravity=reference_gravity,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            scattering_in_emission=self._scattering_in_emission,
            sum_opacities=sum_opacities,
            absorber_present=self.__absorber_present,
            frequencies=self._frequencies,
            weights_gauss=self._lines_loaded_opacities["weights_gauss"],
            cloud_wavelengths=cloud_photosphere_wavelength_boundaries,
            cloud_f_sed=cloud_f_sed,
            cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities=cloud_absorption_opacities,
            photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
        )
        return optical_depths

    def _interpolate_photosphere_radius_profile(
        self,
        pressures,
        optical_depths,
        radius_hydrostatic_equilibrium,
    ):
        pressures = jnp.asarray(pressures, dtype=WORKING_JAX_DTYPE)
        optical_depths = jnp.asarray(optical_depths, dtype=WORKING_JAX_DTYPE)
        radius_hydrostatic_equilibrium = jnp.asarray(
            radius_hydrostatic_equilibrium,
            dtype=FEAUTRIER_JAX_DTYPE,
        )
        tau_target = jnp.asarray(2.0 / 3.0, dtype=WORKING_JAX_DTYPE)
        # The g-point average below must be written as an explicit contraction
        # (einsum), *not* as jnp.sum(weights[:, None, None] * tau, axis=0).
        # The latter is miscompiled by XLA/CPU (jax 0.10.2, Apple silicon) in
        # some of the programs calculate_flux() generates: inside one and the
        # same compiled program, the broadcast-multiply-then-reduce spelling
        # and the contraction spelling return different numbers for identical
        # operands (0.0061 vs. 0.6289 in the case that uncovered this), which
        # then places the tau = 2/3 level tens of layers too deep. Whether the
        # bug triggers depends on the surrounding graph, so a cloudy call and
        # the same cloudy column inside a patchy call disagreed.
        tau_profiles = jnp.einsum(
            "g,gfl->fl",
            self._lines_loaded_opacities["weights_gauss"],
            optical_depths[:, :, 0, :],
        )
        reaches_tau_target = jnp.max(tau_profiles, axis=1) > tau_target

        # Note: the two statements below
        pressure_tau_23 = vmap(
            lambda tau_profile: _interpolate_linear_sorted(
                tau_profile,
                pressures,
                tau_target,
                clip_xquery=True # Prevent extrapolation to not poison gradients.
            )
        )(tau_profiles)
        photosphere_radius = vmap(
            lambda pressure_tau_23_i: _interpolate_linear_sorted(
                pressures,
                radius_hydrostatic_equilibrium,
                pressure_tau_23_i,
                clip_xquery=True # Prevent extrapolation to not poison gradients.
            )
        )(pressure_tau_23)
        # Could be collapsed like this:
        """
        photosphere_radius = vmap(
            lambda tau_profile: _interpolate_linear_sorted(
                tau_profile,
                radius_hydrostatic_equilibrium,
                tau_target,
            )
        )(tau_profiles)
        """

        photosphere_radius = jnp.where(
            reaches_tau_target,
            photosphere_radius,
            radius_hydrostatic_equilibrium[-1],
        )  # Even though we correct values that do not reach tau = 2/3
           # an extrapolation to jnp.inf could "poison" the gradient, that's
           # why we use the clip_xquery = True above.

        def do_nothing():
            return None

        # Claude timed this debug.print and it costs like 10 microseconds per call
        # TODO: think of a better way to warn user. Return error code instead?
        def printing_warning():
            debug.print(
                'Atmosphere is too transparent for calculating the "photospheric '
                'radius" at tau = 2/3. Returning the radius at maximum pressure instead!'
            )

        lax.cond(jnp.all(reaches_tau_target),
                 do_nothing,
                 printing_warning)

        return jnp.asarray(photosphere_radius, dtype=FEAUTRIER_JAX_DTYPE)

    @jit(
        static_argnames=[
            "cloud_particle_radius_distribution",
            "patchy_clouds",
            "additional_absorption_opacities_function",
            "additional_scattering_opacities_function",
            "return_opacities",
            "sum_opacities",
            "anisotropic_cloud_scattering",
            "compute_clear_opacities",
            "ck_chunk_size",
            "make_analytical_clouds_patchy",
        ]
    )
    def _calculate_opacities(
        self,
        pressures,
        temperatures,
        mass_fractions,
        mean_molar_masses,
        reference_gravity, # Needed to calculate cloud radii if not externally supplied
                           # or to calculate the cloud scaling at the clear photosphere,
                           # if requested.
        opaque_cloud_top_pressure=None,
        cloud_particle_mean_radii=None,
        cloud_particle_radius_distribution_std=None, # TODO: add Roberto's fix for small sigma_g
        cloud_particle_radius_distribution="lognormal",
        cloud_particle_number_density_grid=None,
        cloud_hansen_a=None,
        cloud_hansen_b=None,
        clouds_particle_porosity_factor=None,
        cloud_f_sed=None,
        eddy_diffusion_coefficients=None,
        haze_factor=1.0,
        power_law_opacity_350nm=None,
        power_law_opacity_coefficient=None,
        gray_opacity=None,
        cloud_photosphere_median_optical_depth=None,
        patchy_clouds=None,
        return_cloud_contribution=False,
        return_opacities=False,
        additional_absorption_opacities_function=_additional_opacities_placeholder_fn,
        additional_scattering_opacities_function=_additional_opacities_placeholder_fn,
        compute_clear_opacities=False,
        sum_opacities=False,
        anisotropic_cloud_scattering=None,
        target_particle_radii=None, # TODO Annie: decide on more descriptive name (e.g., target_particle_radii_onoff)
        target_pressure=None,
        ck_chunk_size=radiative_transfer_core.DEFAULT_CK_CHUNK_SIZE,
        make_analytical_clouds_patchy=False,
        additional_opacity_functions_kwargs=None,
    ):
        """Build line, continuum, and cloud opacity for spectral calculations.

        The method interpolates line opacities to the current atmospheric
        structure (P and T), computes gas and cloud continuum contributions, combines
        them according to the active line-opacity mode (correlated-k in scattering or
        not in scattering mode, line-by-line calculations).

        Args:
            temperatures:
                Atmospheric temperature profile in K. Shape (N_layers).
            mass_fractions:
                Dictionary containing mass-fraction profiles for all atmospheric species
                that have a requested opacity associated to them. Shape: dictionary with
                shape (N_layers) array per species inside.
            mean_molar_masses:
                Mean molecular mass profile in amu. Shape (N_layers).
            reference_gravity:
                Reference gravity in cgs units. Scalar.
            opaque_cloud_top_pressure:
                Optional pressure in bar where an opaque cloud deck is imposed. Scalar.
            cloud_particle_mean_radii:
                Optional cloud particle mean radii keyed by cloud species.
                Shape: dictionary with shape (N_layers) array per species inside.
            cloud_particle_radius_distribution_std:
                Width of the log-normal particle size distribution. Scalar, or array of
                shape (N_layers), or dictionary with scalar or shape (N_layers) array per species inside.
            cloud_particle_radius_distribution:
                Cloud particle size distribution model, typically
                ``"lognormal"`` or ``"hansen"``.
            cloud_particle_number_density_grid: # Todo Annie: fix name (dndr?) and add dimensions
                Optional per-species particle number density grid.
            cloud_hansen_a:
                Optional Hansen ``a`` parameter profiles keyed by cloud species.
                Scalar, or array of shape (N_layers), or dictionary with scalar
                or shape (N_layers) array per species inside.
            cloud_hansen_b:
                Optional Hansen ``b`` parameter profiles keyed by cloud species.
                Scalar, or array of shape (N_layers), or dictionary with scalar
                or shape (N_layers) array per species inside.
            clouds_particle_porosity_factor:
                Optional porosity factors keyed by cloud species (dictionary of scalars).
            cloud_f_sed:
                Cloud settling parameter, either scalar or array or dictionary with
                clouds species names as keys containing scalars or arrays.
            eddy_diffusion_coefficients:
                Optional eddy diffusion coefficient profile. Shape: N_layers
            haze_factor:
                Scalar multiplier for Rayleigh scattering.
            power_law_opacity_350nm:
                Optional reference scattering opacity at 0.35 micron.
            power_law_opacity_coefficient:
                Wavelength exponent for the parametrized scattering opacity.
            gray_opacity:
                Optional gray absorption opacity added "everywhere"
                (at all pressures and wavelengths). Scalar.
            cloud_photosphere_median_optical_depth:
                Optional target median cloud optical depth near the photosphere.
                Scalar.
            patchy_clouds:
                Cloud species affected by ``cloud_fraction`` and therefore
                excluded from the always-cloudy branch.
            return_cloud_contribution:
                If ``True``, propagate cloud contribution products required by
                downstream diagnostics.
            additional_absorption_opacities_function:
                Optional user-supplied absorption opacity function.
            additional_scattering_opacities_function:
                Optional user-supplied scattering opacity function.
            compute_clear_opacities:
                If ``True``, also construct clear-column opacities.
            target_particle_radii:
                Particle radius to target for selective bin disabling. Radius bin closest
                the input value will be disabled.
            target_pressure:
                Pressure to target for selective bin disabling. Pressure bin closest to
                the input value will be disabled.
            make_analytical_clouds_patchy:
                If true, the opaque cloud and powerlaw cloud are treated as patchy cloud species,
                if requested.
            additional_opacity_functions_kwargs:
                Optional dictionary of extra keyword arguments handed to both
                ``additional_absorption_opacities_function`` and
                ``additional_scattering_opacities_function``.

        Returns:
            A tuple ``(opacities, continuum_opacities_scattering,
            cloud_scattering_opacities, cloud_absorption_opacities,
            cloud_opacities, cloud_particle_mean_radii,
            cloud_scattering_in_transmission, clear_opacities,
            clear_continuum_opacities_scattering,
            clear_cloud_scattering_opacities, clear_cloud_absorption_opacities)``:

            opacities:
                The interpolated line opacities, weighted by their mass
                fractions, with the continuum absorption opacities added
                into the first species slot. If ``sum_opacities`` is
                ``True`` (in c-k mode, this only happens in emission with
                ``scattering_in_emission=True``; in line-by-line mode this
                is always the case), the species are further combined into
                that first slot: via the correlated-k
                mixing procedure in c-k mode, or by a plain sum in
                frequency space in line-by-line mode. The total combined
                opacity then occupies the first (and, in c-k mode, only)
                species slot. Otherwise (c-k mode without
                ``sum_opacities``), species are left separate along the
                species axis, with only the plain continuum opacity added
                to the first slot. Shape
                ``(N_g, N_frequencies, N_species, N_layers)``, where
                ``N_g`` is 1 in line-by-line mode, and ``N_species`` is 1
                whenever the species axis has been collapsed (c-k with
                ``sum_opacities=True``, or line-by-line with more than one
                line species, ``sum_opacities=True``, and
                ``return_opacities=False``).
            continuum_opacities_scattering:
                The combined gas and cloud continuum scattering opacities
                (the cloud part optionally (1-g)-scaled, if
                ``anisotropic_cloud_scattering`` is ``True``), including the
                contribution of patchy clouds. Shape
                ``(N_frequencies, N_layers)``.
            cloud_scattering_opacities:
                The total cloud *scattering* opacities, (1-g)-scaled.
                Shape ``(N_frequencies, N_layers)``.
            cloud_absorption_opacities:
                The total cloud *absorption* opacities.
                Shape ``(N_frequencies, N_layers)``.
            cloud_opacities:
                The total cloud *extinction* opacities, i.e., absorption
                plus the (1-g)-scaled scattering opacity.
                Shape ``(N_frequencies, N_layers)``.
            cloud_particle_mean_radii:
                The cloud particle mean radii actually used (as supplied,
                or as derived from ``cloud_f_sed`` and
                ``eddy_diffusion_coefficients`` when not supplied), per
                cloud species and atmospheric layer.
                Shape ``(N_layers, N_clouds)``.
            cloud_scattering_in_transmission:
                Scalar boolean flag, ``True`` if scattering must be
                accounted for in the transmission spectrum, i.e., if
                ``rayleigh_species`` is non-empty, or if any cloud species
                (in the cloudy or, if computed, the clear branch) has a
                non-zero mass fraction anywhere in the atmosphere. Note
                that neither of the two current callers of
                ``_calculate_opacities`` (:meth:`_calculate_flux_kernel`
                and :meth:`_calculate_transit_radii_full_kernel`) actually
                use this returned value.
            clear_opacities:
                Same as ``opacities``, but neglecting the contribution of
                patchy clouds: the ``patchy_clouds`` species, and, if
                ``make_analytical_clouds_patchy`` is ``True``, the opaque
                cloud deck and power-law opacity as well. Shape
                ``(N_g, N_frequencies, N_species, N_layers)``. Zeros
                (``jnp.zeros_like(opacities)``) if ``compute_clear_opacities``
                is ``False``.
            clear_continuum_opacities_scattering:
                Same as ``continuum_opacities_scattering``, but neglecting
                the contribution of patchy clouds. Zeros if
                ``compute_clear_opacities`` is ``False``.
                Shape ``(N_frequencies, N_layers)``.
            clear_cloud_scattering_opacities:
                Same as ``cloud_scattering_opacities``, but neglecting the
                contribution of patchy clouds.
                Shape ``(N_frequencies, N_layers)``.
            clear_cloud_absorption_opacities:
                Same as ``cloud_absorption_opacities``, but neglecting the
                contribution of patchy clouds.
                Shape ``(N_frequencies, N_layers)``.
        """

        if patchy_clouds is None:
            patchy_clouds = tuple()
        else:
            patchy_clouds = tuple(patchy_clouds)

        if anisotropic_cloud_scattering is None: # TODO Paul_review: add to docstring once it is clear what it does
                                                 # Likely if cloud opacities are multiplied by (1-g)?
            anisotropic_cloud_scattering = self._anisotropic_cloud_scattering

        if anisotropic_cloud_scattering == "auto":
            anisotropic_cloud_scattering = self._scattering_in_emission

        # Fill line mass fraction dictionary with provided mass fraction dictionary
        if len(self._line_species) > 0:
            line_species_mass_fraction_columns = []

            for species in self._line_species:
                # Check if user provided the detailed line absorber name or if line absorber name should resolved
                # assuming a default species.
                # The looping works  because self._line_species is static (aux_data in _tree_flatten) and while
                # mass_fractions is traced, its dictionary keys are always static.
                if species in mass_fractions:
                    line_species_mass_fraction_columns.append(
                        jnp.asarray(mass_fractions[species], dtype=WORKING_JAX_DTYPE)
                    )
                    continue

                default_species, spectral_info = Opacity.split_species_spectral_info(
                    species
                )

                if default_species not in mass_fractions:
                    raise ValueError(
                        f"line species '{species}' not found in mass fractions dict "
                        f"(listed species: {list(mass_fractions.keys())}"
                    )

                line_species_mass_fraction_columns.append(
                    jnp.asarray(
                        mass_fractions[default_species], dtype=WORKING_JAX_DTYPE
                    )
                )

            line_species_mass_fractions = jnp.stack(
                line_species_mass_fraction_columns, axis=1
            )

        else:
            # If there are no specified line species then we need at least an array to contain the continuum opacities
            line_species_mass_fractions = jnp.zeros(
                (pressures.size, 1), dtype=WORKING_JAX_DTYPE
            )

        # Continuum opacities, absorption only (currently CIA of various species and H- b-f and H- f-f)
        # TODO Paul_review: look inside
        continuum_opacities = continuum.compute_continuum_opacities(
            frequencies=self._frequencies,
            wavelengths_micron=self._wavelengths_micron,
            frequency_bins_edges=self._frequency_bins_edges,
            pressures=pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            cias_loaded_opacities=self._cias_loaded_opacities,
            gas_continuum_contributors=self._gas_continuum_contributors,
            gray_opacity=gray_opacity,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_opacity_functions_kwargs=additional_opacity_functions_kwargs,
            # species_basenames=self._species_basenames # TODO remove this commented line?
        )
        # TODO: discuss and remove the whole "scattering_in_transmission" stuff?
        # Continuum opacities, scattering only (currently Rayleigh scattering of gas, only)
        # TODO Paul_review: look inside
        continuum_opacities_scattering, scattering_in_transmission = (
            continuum.compute_continuum_scattering_opacities(
                frequencies=self._frequencies,
                wavelengths_micron=self._wavelengths_micron,
                pressures=pressures,
                temperatures=temperatures,
                mass_fractions=mass_fractions,
                mean_molar_masses=mean_molar_masses,
                rayleigh_species=self._rayleigh_species,
                haze_factor=haze_factor,
                additional_scattering_opacities_function=additional_scattering_opacities_function,
                additional_opacity_functions_kwargs=additional_opacity_functions_kwargs,
                # species_basenames=self._species_basenames # TODO remove this commented line?
            )
        )

        # TODO Annie review: look inside
        (
            cloudy_cloud_continuum_opacities, # total cloud *absorption* opacities (may include opaque cloud deck).
                                              # Used in the actual RT calculation!
            cloudy_cloud_continuum_opacities_scattering, # total cloud *scattering* opacities,
                                                         # optionally scaled by (1-g)
                                                         # (may include analytical scattering model).
                                                         # Used in the actual RT calculation!
            cloud_scattering_opacities, # total cloud *scattering* opacities, scaled by (1-g)
            cloud_absorption_opacities, # total cloud *absorption* opacities
            cloud_opacities, # total cloud extinction opacity, so continuum and (1-g) (not optional) *
                             # scattering opacity.
            _cloud_particle_mean_radii,
            clear_cloud_continuum_opacities, # same as cloudy_cloud_continuum_opacities above,
                                             # but neglecting the contribution of patchy clouds.
            clear_cloud_continuum_opacities_scattering, # see comment above
            clear_cloud_scattering_opacities, # see comment above
            clear_cloud_absorption_opacities, # see comment above
            cloud_scattering_in_transmission, # TODO: review what this does / if it is needed
        ) = clouds.compute_cloud_continuum_opacities(
            frequencies=self._frequencies,
            frequency_bins_edges=self._frequency_bins_edges,
            pressures=pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            scattering_in_transmission=scattering_in_transmission,
            sum_opacities=sum_opacities,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            cloud_species=self._cloud_species,
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            clouds_loaded_opacities=self._clouds_loaded_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            cloud_particle_mean_radii=cloud_particle_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            patchy_clouds=patchy_clouds,
            return_cloud_contribution=return_cloud_contribution,
            compute_clear_opacities=compute_clear_opacities,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
            make_analytical_clouds_patchy=make_analytical_clouds_patchy,
        )


        # Line opacities (calculating them just before calculating the combined opacities reduces peak memory usage)
        ck_flattened_layout = self._line_opacity_mode == "c-k" and sum_opacities
        collapse_lbl_species_axis = (
            self._line_opacity_mode == "lbl"
            and sum_opacities
            and not return_opacities
            and len(self._line_species) > 1
        )

        if collapse_lbl_species_axis:
            # Interpolate line opacities to requested pressure and temperature,
            # add them in frequency space
            # TODO Paul_review: look inside
            line_opacities = interpolate_species_opacities_summed(
                pressures=pressures,
                temperatures=temperatures,
                n_g=self._lines_loaded_opacities["g_gauss"].size,
                n_frequencies=self._frequencies.size,
                line_opacities_grid=self._lines_loaded_opacities[
                    "opacity_grid_stacked"
                ],
                line_opacities_temperature_pressure_grid=self._lines_loaded_opacities[
                    "temperature_pressure_grid_stacked"
                ],
                line_opacities_temperature_grid_size=self._lines_loaded_opacities[
                    "temperature_grid_size_array"
                ],
                line_opacities_pressure_grid_size=self._lines_loaded_opacities[
                    "pressure_grid_size_array"
                ],
                line_species_mass_fractions=line_species_mass_fractions,
            )
            combined_line_species_mass_fractions = jnp.ones(
                (pressures.size, 1), dtype=WORKING_JAX_DTYPE
            )
        else:
            # Interpolate line opacities to requested pressure and temperature,
            # retain them separately per species (to c-k combine them later below)
            # TODO Paul_review: look inside
            line_opacities = interpolate_species_opacities(
                pressures=pressures,
                temperatures=temperatures,
                n_g=self._lines_loaded_opacities["g_gauss"].size,
                n_frequencies=self._frequencies.size,
                line_opacities_grid=self._lines_loaded_opacities[
                    "opacity_grid_stacked"
                ],
                line_opacities_temperature_pressure_grid=self._lines_loaded_opacities[
                    "temperature_pressure_grid_stacked"
                ],
                line_opacities_temperature_grid_size=self._lines_loaded_opacities[
                    "temperature_grid_size_array"
                ],
                line_opacities_pressure_grid_size=self._lines_loaded_opacities[
                    "pressure_grid_size_array"
                ],
                output_layout="fl_s_ng" if ck_flattened_layout else "ng_f_s_l",
            )
            combined_line_species_mass_fractions = line_species_mass_fractions

        # Combine the continuum absorption opacities (gas and clouds)
        cloudy_continuum_opacities = (
            continuum_opacities + cloudy_cloud_continuum_opacities
        )

        zero_continuum_for_combination = jnp.zeros_like(cloudy_continuum_opacities)

        # Always return the total scattering continuum so transmission and other
        # callers can consume cloud scattering even when emission scattering is disabled.
        # The microphysical clouds are optionally scaled by (1-g),
        # depending on the requested anisotropic_cloud_scattering mode.
        cloudy_continuum_opacities_scattering = (
            continuum_opacities_scattering + cloudy_cloud_continuum_opacities_scattering
        )

        # Combine line opacities
        # If opacities are summed / c-k combined, the total combined absorption
        # opacity is found in the first line species slot.
        # No continuum opacities are actually added here yet (see
        # continuum_opacities=zero_continuum_for_combination), since
        # we want the same line combination to be used for clear and cloudy
        # opacities
        # TODO Paul_review look inside
        line_mix_opacities = continuum.compute_combined_opacities(  # TODO: weird that this is in "continuum."
            opacities=line_opacities,
            continuum_opacities=zero_continuum_for_combination,  # clear_continuum_opacities,
            line_species=self._line_species,
            line_species_mass_fractions=combined_line_species_mass_fractions,
            line_opacity_mode=self._line_opacity_mode,
            sum_opacities=sum_opacities,
            g_gauss=self._lines_loaded_opacities["g_gauss"],
            weights_gauss=self._lines_loaded_opacities["weights_gauss"],
            g_combination_weights=self._lines_loaded_opacities[
                "ck_g_combination_weights"
            ],
            g_combination_weights_sum=self._lines_loaded_opacities[
                "ck_g_combination_weights_sum"
            ],
            input_layout="fl_s_ng" if ck_flattened_layout else "ng_f_s_l",
            ck_chunk_size=ck_chunk_size,
        )

        # Add cloudy continuum absorption opacities
        opacities = line_mix_opacities.at[:, :, 0, :].set(
            line_mix_opacities[:, :, 0, :] + cloudy_continuum_opacities[None, :, :]
        )

        if compute_clear_opacities:

            # Absorption only continuum opacities for "clear" case.
            # For the clouds the contribution
            # of patchy species is neglected.
            clear_continuum_opacities = (
                continuum_opacities + clear_cloud_continuum_opacities
            )
            # Scattering only continuum opacities for "clear" case.
            # For the clouds the contribution
            # of patchy species is neglected.
            clear_continuum_opacities_scattering = (
                continuum_opacities_scattering
                + clear_cloud_continuum_opacities_scattering
            )

            # Add continuum absorption opacities for the clear case
            clear_opacities = line_mix_opacities.at[:,:,0,:].set(
                line_mix_opacities[:, :, 0, :] + clear_continuum_opacities[None, :, :]
            )

        else:

            clear_opacities = jnp.zeros_like(opacities)
            clear_continuum_opacities_scattering = jnp.zeros_like(
                continuum_opacities_scattering
            )

        return (
            opacities,
            cloudy_continuum_opacities_scattering,
            cloud_scattering_opacities,
            cloud_absorption_opacities,
            cloud_opacities,
            _cloud_particle_mean_radii,
            cloud_scattering_in_transmission,
            clear_opacities,
            clear_continuum_opacities_scattering,
            clear_cloud_scattering_opacities,
            clear_cloud_absorption_opacities,
        )

    # TODO Nora review
    def _calculate_transit_radii(
        self,
        pressures,
        temperatures,
        mean_molar_masses,
        reference_gravity,
        reference_pressure,
        planet_radius,
        variable_gravity,
        opacities,
        continuum_opacities_scattering,
        sum_opacities,
        return_contributions,
    ):
        """Compute transmission-spectrum radii for the prepared opacity state.

        Args:
            pressures:
                Pressure at each atmospheric level in cgs units.
            temperatures:
                Atmospheric temperature profile in K.
            mean_molar_masses:
                Mean molecular mass profile in amu.
            reference_gravity:
                Reference gravity in cgs units at ``reference_pressure``.
            reference_pressure:
                Pressure in bar where ``planet_radius`` is defined.
            planet_radius:
                Planet radius in cm at ``reference_pressure``.
            variable_gravity:
                If ``True``, use a $1/r^2$ gravity profile in the transmission
                calculation.
            opacities:
                Total opacity array produced by ``_calculate_opacities``.
            continuum_opacities_scattering:
                Continuum scattering opacity array.
            sum_opacities:
                Whether the main transmission spectrum should use the summed
                opacity contract.
            return_contributions:
                Whether to evaluate the transmission contribution function on
                the same opacity contract as the main transmission spectrum.

        Returns:
            Tuple ``(transit_radii, radius_hydrostatic_equilibrium,
            transmission_contribution)``.
        """
        transmission_contribution = jnp.zeros(
            (jnp.size(pressures), self._frequencies.size), dtype=FEAUTRIER_JAX_DTYPE
        )

        # With summed opacities (line-by-line, several species) the total opacity
        # sits in the first species slot. The remaining slots still hold the
        # per-species opacities whenever the species axis was not collapsed
        # upstream (e.g. return_opacities=True), so only slot 0 may enter the
        # transmission product -- otherwise every species is counted twice.
        if sum_opacities and len(self._line_species) > 1:
            transit_opacities = opacities[:, :, :1, :]
        else:
            transit_opacities = opacities

        # Calculate the transmission spectrum
        transit_radii, radius_hydrostatic_equilibrium = (
            radiative_transfer_core.compute_transit_radii(
                opacities=transit_opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                pressures=pressures * 1e-6,  # cgs to bar
                temperatures=temperatures,
                weights_gauss=self._lines_loaded_opacities["weights_gauss"],
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                reference_pressure=reference_pressure,
                planet_radius=planet_radius,
                variable_gravity=variable_gravity,
            )
        )
        if return_contributions:
            transmission_contribution, radius_hydrostatic_equilibrium = (
                radiative_transfer_core.compute_transmission_spectrum_contribution(
                    opacities=opacities,
                    temperatures=temperatures,
                    pressures=pressures * 1e-6,  # cgs to bar
                    reference_gravity=reference_gravity,
                    mean_molar_masses=mean_molar_masses,
                    reference_pressure=reference_pressure,
                    planet_radius=planet_radius,
                    weights_gauss=self._lines_loaded_opacities["weights_gauss"],
                    scattering_in_transmission=True,
                    continuum_opacities_scattering=continuum_opacities_scattering,
                    variable_gravity=variable_gravity,
                )
            )
        return transit_radii, radius_hydrostatic_equilibrium, transmission_contribution

    def _init_frequency_grid(self):
        import h5py

        """Initialize the Radtrans frequency grid, used to calculate the spectra.
        The frequency grid comes from the requested opacity files, in the following priority order:
            1. lines,
            2. CIA,
            3. clouds.
        If no opacities are provided, the mean of the wavelength boundaries is used. The frequency grid in that case
        has only 1 element.

        Returns:
            frequencies:
                (Hz) frequencies (center of bin) of the line opacities, also use for spectral calculations, of size N
            frequency_bins_edges:
                (Hz) edges of the frequencies bins, of size N+1
                for correlated-k only, number of points used to sample the g-space (1 in the case lbl is used)
        """
        if len(self._line_species) > 0:
            frequencies, frequency_bins_edges = self._init_frequency_grid_from_lines()
        elif len(cia.get_cia_contributors(self._gas_continuum_contributors)) > 0:
            hdf5_file = cia.get_cia_aliases(cia.get_cia_contributors(self._gas_continuum_contributors)[0])
            hdf5_file = CIAOpacity.find(
                path_input_data=self._path_input_data, species=hdf5_file
            )

            with h5py.File(hdf5_file, "r") as f:
                frequency_grid = cst.c * f["wavenumbers"][:]  # cm-1 to Hz

            frequencies, frequency_bins_edges = (
                self._init_frequency_grid_from_frequency_grid(
                    frequency_grid=frequency_grid,
                    wavelength_boundaries=self._wavelength_boundaries,
                    sampling=1,
                )
            )
        elif len(self._cloud_species) > 0:
            hdf5_file = CloudOpacity.get_aliases(self._cloud_species[0])
            hdf5_file = CloudOpacity.find(
                path_input_data=self._path_input_data, species=hdf5_file
            )

            with h5py.File(hdf5_file, "r") as f:
                frequency_grid = cst.c * f["wavenumbers"][:]  # cm-1 to Hz

            frequencies, frequency_bins_edges = (
                self._init_frequency_grid_from_frequency_grid(
                    frequency_grid=frequency_grid,
                    wavelength_boundaries=self._wavelength_boundaries,
                    sampling=1,
                )
            )
        else:
            debug.print(
                "no opacity source given (lines, CIA, or clouds), "
                "setting frequency grid using the mean of wavelength boundaries (1 element)"
            )
            frequency_bins_edges = jnp.zeros(2, dtype=WORKING_JAX_DTYPE)
            frequency_bins_edges = frequency_bins_edges.at[0].set(
                wavelength2frequency(self._wavelength_boundaries[1] * 1e-4)
            )  # um to cm
            frequency_bins_edges = frequency_bins_edges.at[1].set(
                wavelength2frequency(self._wavelength_boundaries[0] * 1e-4)
            )  # um to cm
            frequencies = jnp.mean(frequency_bins_edges)

        return frequencies, frequency_bins_edges

    @staticmethod
    def _init_frequency_grid_from_frequency_grid(
        frequency_grid, wavelength_boundaries, sampling=1
    ):
        """Returns that subsection of a frequency grid that falls between (inclusive) the
        to wavelength boundaries given (which will be converted to frequencies), accounting
        for opacity downsampling at high resolution.

        Args:
            frequency_grid (jnp arrary of floats):
                frequency grid to be used for constructing the pRT official frequency grid.
            wavelength_boundaries (tuple of floats):
                Wavelength boundaries requested by the user, in microns.
            sampling (integer):
                Whether a frequency sampling is to be applied.
                E.g., sampling = 4 only keeps every 4th frequency point.

        Returns:
            Array of frequencies and the frequency bin edges of said array.
        """
        # Get frequency boundaries
        frequency_min = um2hz(wavelength_boundaries[1])
        frequency_max = um2hz(wavelength_boundaries[0])

        # Check if the requested wavelengths boundaries are within the file boundaries
        bad_boundaries = False

        if frequency_min < frequency_grid[0]:
            bad_boundaries = True

        if frequency_max > frequency_grid[-1]:
            bad_boundaries = True

        if bad_boundaries:
            raise ValueError(
                f"Requested wavelength interval "
                f"({wavelength_boundaries[0]}--{wavelength_boundaries[1]}) "
                f"is out of opacities table wavelength grid "
                f"({hz2um(frequency_grid[-1])}--{hz2um(frequency_grid[0])})"
            )

        # Get the freq. corresponding to the requested boundaries, with the request fully within the selection
        selection = jnp.nonzero(
            jnp.logical_and(
                jnp.greater_equal(frequency_grid, frequency_min),
                jnp.less_equal(frequency_grid, frequency_max),
            )
        )[0]
        selection = jnp.array([selection[0], selection[-1]])

        if frequency_grid[selection[0]] > frequency_min:
            selection = selection.at[0].add(-1)

        if sampling > 1:
            # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
            selection = selection.at[0].subtract(sampling - 1)

        frequencies = frequency_grid[selection[0] : selection[-1] + 1]
        frequencies = frequencies[::-1]

        # Down-sample frequency grid in lbl mode if requested
        if sampling > 1:
            frequencies = frequencies[::sampling]

        frequency_bins_edges = jnp.array(
            wavelength2frequency(compute_bins_edges(frequency2wavelength(frequencies))),
            dtype=WORKING_JAX_DTYPE,
        )

        return frequencies, frequency_bins_edges

    def _init_frequency_grid_from_lines(self):
        """Loads opacity files from the first requested line species to define the frequency grid within the
        requested wavelength range. The species wavelength range is expanded if smaller than the user request,
        and cut down if smaller.

        Returns:
            Frequencies and frequency bin edges of the defined array.
        """
        import h5py

        if self._line_opacity_mode == "c-k":  # correlated-k
            # Get dimensions of molecular opacity arrays for a given P-T point, they define the resolution
            # Use the first entry of self.line_species for this, if given
            opacities_file = load.get_line_opacity_file(
                path_input_data=self._path_input_data,
                species=self._line_species[0],
                category="correlated_k_opacities",
            )

            with h5py.File(opacities_file, "r") as f:
                frequency_bins_edges = cst.c * f["bin_edges"][:][::-1]

            wavelengths = hz2um(frequency_bins_edges)
        elif self._line_opacity_mode == "lbl":  # line-by-line
            # Load the wavelength grid
            opacities_file = LineByLineOpacity.find(
                path_input_data=self._path_input_data, species=self._line_species[0]
            )

            with h5py.File(opacities_file, "r") as f:
                frequency_grid = cst.c * f["bin_edges"][:]  # cm-1 to Hz

            if (
                frequency_grid[0] == frequency_grid[1]
            ):  # handle duplicated last wavenumber
                frequency_grid = frequency_grid[1:]

            wavelengths = hz2um(frequency_grid[::-1])
        else:
            raise ValueError(
                f"line opacity mode must be 'c-k' or 'lbl', but was '{self._line_opacity_mode}'"
            )

        # Extend the wavelength range if user requests larger range than what first line opa species contains
        if wavelengths[-1] < self._wavelength_boundaries[1]:
            delta_log_wavelength = jnp.diff(jnp.log10(wavelengths))[-1]

            add_high = (
                10
                ** jnp.arange(
                    jnp.log10(wavelengths[-1]),
                    jnp.log10(self._wavelength_boundaries[-1]) + delta_log_wavelength,
                    delta_log_wavelength,
                )[1:]
            )
            wavelengths = jnp.concatenate((wavelengths, add_high))

        if wavelengths[0] > self._wavelength_boundaries[0]:
            delta_log_wavelength = jnp.diff(jnp.log10(wavelengths))[0]

            if delta_log_wavelength == 0:  # handle duplicate first wavelength
                delta_log_wavelength = jnp.diff(jnp.log10(wavelengths))[1]

            add_low = 1e1 ** (
                -np.arange(
                    -jnp.log10(wavelengths[0]),
                    -jnp.log10(self._wavelength_boundaries[0]) + delta_log_wavelength,
                    delta_log_wavelength,
                )[1:][::-1]
            )
            wavelengths = jnp.concatenate((add_low, wavelengths))

        # Get back to frequencies
        if self._line_opacity_mode == "c-k":
            frequency_bins_edges = um2hz(wavelengths)
            frequencies = (frequency_bins_edges[1:] + frequency_bins_edges[:-1]) * 0.5

            # Cut the wavelength range if user requests smaller range than what first line opa species contains
            indices_within_boundaries = jnp.nonzero(
                jnp.logical_and(
                    jnp.greater(
                        frequency2wavelength(frequencies),
                        self._wavelength_boundaries[0] * 1e-4,
                    ),  # um to cm
                    jnp.less(
                        frequency2wavelength(frequencies),
                        self._wavelength_boundaries[1] * 1e-4,
                    ),  # um to cm
                )
            )[0]

            frequencies = jnp.array(
                frequencies[indices_within_boundaries], dtype=WORKING_JAX_DTYPE
            )

            # Get the corresponding frequencies bin edges, +2 is to catch the upper bin edge
            # +1 because there's N+1 edges to N frequency points, and another
            # +1 to be larger than larger boundary wavelength, actually.
            frequency_bins_edges = jnp.array(
                frequency_bins_edges[
                    indices_within_boundaries[0] : indices_within_boundaries[-1] + 2
                ],
                dtype=WORKING_JAX_DTYPE,
            )
        elif self._line_opacity_mode == "lbl":
            frequency_grid = um2hz(wavelengths[::-1])
            frequencies, frequency_bins_edges = (
                self._init_frequency_grid_from_frequency_grid(
                    frequency_grid=frequency_grid,
                    wavelength_boundaries=self._wavelength_boundaries,
                    sampling=self._line_by_line_opacity_sampling,
                )
            )
        else:
            raise ValueError(
                f"line opacity mode must be 'c-k' or 'lbl', but was '{self._line_opacity_mode}'"
            )

        return frequencies, frequency_bins_edges

    def _tree_flatten(self):
        cia_children = tuple(
            (
                collision_dict["weight"],
                collision_dict["lambda"],
                collision_dict["temperature"],
                collision_dict["alpha"],
            )
            for collision_dict in self._cias_loaded_opacities.values()
        )
        children = (
            self._pressures,
            self._wavelength_boundaries,
            self._frequencies,
            self._wavelengths_micron,
            self._frequency_bins_edges,
            self._lines_loaded_opacities,
            cia_children,
            self._clouds_loaded_opacities,
            self._emission_angle_grid,
        )
        aux_data = {
            "line_species": tuple(self._line_species),
            "gas_continuum_contributors": tuple(self._gas_continuum_contributors),
            "rayleigh_species": tuple(self._rayleigh_species),
            "cloud_species": tuple(self._cloud_species),
            "line_opacity_mode": self._line_opacity_mode,
            "line_by_line_opacity_sampling": self._line_by_line_opacity_sampling,
            "scattering_in_emission": self._scattering_in_emission,
            "anisotropic_cloud_scattering": self._anisotropic_cloud_scattering,
            "retain_unstacked_line_opacities_for_plotting": self._retain_unstacked_line_opacities_for_plotting,
            "path_input_data": self._path_input_data,
            "cia_collision_names": tuple(self._cias_loaded_opacities.keys()),
            "cia_collision_molecules": tuple(
                None
                if collision_dict["molecules"] is None
                else tuple(collision_dict["molecules"])
                for collision_dict in self._cias_loaded_opacities.values()
            ),
            "sum_opacities": self.__sum_opacities,
            "absorber_present": self.__absorber_present,
            "sum_opacities_emission": self._sum_opacities_emission,
            "anisotropic_cloud_scattering_emission": self._anisotropic_cloud_scattering_emission,
        }
        # A tuple of items keeps the treedef hashable, so jit dispatch uses hash
        # lookups instead of falling back to per-call dict equality comparisons
        # This is needed to make cashe lookups fast or even enable them in the first place.
        return children, tuple(sorted(aux_data.items()))

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        aux_data = dict(aux_data)
        instance = cls.__new__(cls)
        (
            instance._pressures,
            instance._wavelength_boundaries,
            instance._frequencies,
            instance._wavelengths_micron,
            instance._frequency_bins_edges,
            instance._lines_loaded_opacities,
            cia_children,
            instance._clouds_loaded_opacities,
            instance._emission_angle_grid,
        ) = children
        cia_loaded_opacities = LockedDict()
        for collision_name, molecules, collision_children in zip(
            aux_data["cia_collision_names"],
            aux_data["cia_collision_molecules"],
            cia_children,
        ):
            weight, wavelengths, temperatures, alpha = collision_children
            cia_loaded_opacities[collision_name] = LockedDict.build_and_lock(
                {
                    "molecules": molecules,
                    "weight": weight,
                    "lambda": wavelengths,
                    "temperature": temperatures,
                    "alpha": alpha,
                }
            )
        cia_loaded_opacities.lock()
        instance._cias_loaded_opacities = cia_loaded_opacities
        instance._line_species = aux_data["line_species"]
        instance._gas_continuum_contributors = aux_data["gas_continuum_contributors"]
        instance._rayleigh_species = aux_data["rayleigh_species"]
        instance._cloud_species = aux_data["cloud_species"]
        instance._line_opacity_mode = aux_data["line_opacity_mode"]
        instance._line_by_line_opacity_sampling = aux_data[
            "line_by_line_opacity_sampling"
        ]
        instance._scattering_in_emission = aux_data["scattering_in_emission"]
        instance._anisotropic_cloud_scattering = aux_data[
            "anisotropic_cloud_scattering"
        ]
        instance._retain_unstacked_line_opacities_for_plotting = aux_data[
            "retain_unstacked_line_opacities_for_plotting"
        ]
        instance._path_input_data = aux_data["path_input_data"]
        instance._Radtrans__sum_opacities = aux_data["sum_opacities"]
        instance._Radtrans__absorber_present = aux_data["absorber_present"]
        instance._sum_opacities_emission = aux_data["sum_opacities_emission"]
        instance._anisotropic_cloud_scattering_emission = aux_data[
            "anisotropic_cloud_scattering_emission"
        ]
        return instance

    def calculate_opacity_contribution_spectra(
        self,
        mode,
        opaque_cloud_top_pressure=None,
        power_law_opacity_350nm=None,
        power_law_opacity_coefficient=None,
        gray_opacity=None,
        cloud_photosphere_median_optical_depth=None,
        additional_absorption_opacities_function=None,
        additional_scattering_opacities_function=None,
        stellar_intensities=None,
        **kwargs,
    ):
        """Calculate total and per-opacity-source spectra for attribution studies.

        The method evaluates the requested spectral mode once with all active
        opacity sources and then repeats the calculation for each individual
        initialization-time or runtime opacity source, returning a dictionary of
        spectra keyed by contribution name. This is most useful for transmission
        spectra, but can be a bit confusing for emission spectra, since the flux
        when only a minor species is present and all others have been turned off
        can look very different when compared to the full spectrum. If one looks
        at a situation where the flux actually shows high opacities as emission features
        (i.e., due to an atmospheric temperature inversion), it can be quite useful though,
        similar to the transmission spectrum contribution.
        Caution! This function only works as intended if at least one line species is being used, at
        the moment.

        Args:
            mode:
                Either ``"emission"`` or ``"transmission"``.
            opaque_cloud_top_pressure:
                Optional opaque cloud deck pressure in bar.
            power_law_opacity_350nm:
                Optional reference scattering opacity for the parametrized power
                law cloud model.
            power_law_opacity_coefficient:
                Wavelength exponent for the optional parametrized scattering opacity.
            gray_opacity:
                Optional gray absorption opacity.
            cloud_photosphere_median_optical_depth:
                Optional target cloud optical depth near the cloud-free photosphere.
            additional_absorption_opacities_function:
                Optional user-provided absorption opacity callback.
            additional_scattering_opacities_function:
                Optional user-provided scattering opacity callback.
            stellar_intensities:
                Optional stellar intensity spectrum used in emission mode.
            **kwargs:
                Additional keyword arguments forwarded to
                ``calculate_flux`` or ``calculate_transit_radii``.

        Returns:
            Nested dictionary containing the total spectrum under ``"Total"``
            and opacity contribution spectra grouped by opacity source.
        """
        # Store opacity sources values
        opacity_sources = load.get_opacity_sources_dict() # This is empty by default!
        opacity_sources.update(
            {
                "init": {
                    "line_species": self.line_species,
                    "gas_continuum_contributors": self.gas_continuum_contributors,
                    "cloud_species": self.cloud_species,
                    "rayleigh_species": self.rayleigh_species,
                },
                "runtime": {
                    "opaque_cloud_top_pressure": opaque_cloud_top_pressure,
                    "power_law": {
                        "power_law_opacity_350nm": power_law_opacity_350nm,
                        "power_law_opacity_coefficient": power_law_opacity_coefficient,
                    },
                    "gray_opacity": gray_opacity,
                    "cloud_photosphere_median_optical_depth": cloud_photosphere_median_optical_depth,
                    "additional_absorption_opacities_function": additional_absorption_opacities_function,
                    "additional_scattering_opacities_function": additional_scattering_opacities_function,
                    "stellar_intensities": stellar_intensities,
                },
            }
        )

        # Default values for opacity sources
        opacity_source_default = load.get_opacity_sources_dict() # This is empty by default!

        # Initialize spectral mode
        if mode == "emission":
            spectral_function = "calculate_flux"
        elif mode == "transmission":
            spectral_function = "calculate_transit_radii"

            # Remove spectral parameters that are not implemented in transmission
            del opacity_sources["runtime"]["cloud_photosphere_median_optical_depth"]
            del opacity_sources["runtime"]["stellar_intensities"]
        else:
            raise ValueError(
                f"mode '{mode}' is not implemented, "
                f"available modes are 'emission'|'transmission'"
            )

        # Initialize opacity contributions dict
        opacity_contributions = {}

        # Total spectrum
        spectral_parameters = { # Contains all keyword args that are needed for the
                                # calculate_flux or calculate_transit_radii calls.
            copy.deepcopy(k): copy.deepcopy(v) for k, v in kwargs.items()
        }

        for key, value in opacity_sources["runtime"].items():
            if key == "power_law":
                spectral_parameters["power_law_opacity_350nm"] = copy.deepcopy(
                    value["power_law_opacity_350nm"]
                )
                spectral_parameters["power_law_opacity_coefficient"] = copy.deepcopy(
                    value["power_law_opacity_coefficient"]
                )
            else:
                spectral_parameters[key] = copy.deepcopy(value)

        # Evaluate the appropriate spectral function with all opacity species
        # and all requested keyword arguments. This is the full (nominal) spectrum
        # Called "total".
        opacity_contributions["Total"] = self.__getattribute__(spectral_function)(
            **spectral_parameters
        )

        # Contribution spectra
        _default_radtrans = None
        _default_mass_fractions = None

        for opacity_type, opacity_source in opacity_sources.items():
            # Branch for "line_species", "gas_continuum_contributors",
            # "rayleigh_species", "cloud_species":
            if opacity_type == "init":  # fixed (at instantiation) opacity sources
                for (
                    opacity, # will be "line_species", "gas_continuum_contributors",
                             # "rayleigh_species", "cloud_species"
                    species_list, # will be the contributing list of species.
                ) in opacity_source.items():  # all fixed opacity sources are lists
                    opacity_contributions[opacity] = {} # make an empty dictionary for the,
                                                        # e.g., "line_species" key.

                    # Init a temporary radtrans object with only the relevant opacity source
                    for species in species_list: # loop through the species
                        _user_cloud_species = None

                        # Use full names for CIA and clouds to have a clean legend when plotting the opacities
                        if opacity == "gas_continuum_contributors":
                            species = cia.get_cia_aliases(species)
                        elif opacity == "cloud_species":
                            _user_cloud_species = species
                            species = CloudOpacity.get_aliases(species)

                        fixed_opacity_sources = opacity_source_default[opacity_type] # since we are
                                                # in the "init" branch, this will return a dictionary
                                                # with keys "line_species", "gas_continuum_contributors",
                                                # "rayleigh_species", "cloud_species", all containing
                                                # empty tuples
                        fixed_opacity_sources[opacity] = [species] # now we overwrite one of these empty
                                                                   # tuples with a list that contains
                                                                   # the one species for which we want
                                                                   # to calculate the spectrum
                                                                   # (will be cast to tuple within pRT)

                        print(
                            f"Generating temporary Radtrans object with '{opacity}': '{species}'"
                        )

                        # Instantiate a default Radtrans object that will be used for runtime opacity sources
                        if _default_radtrans is None: # this happens just once, after that it isn't None any longer
                            _default_radtrans = Radtrans( # This mirrors the pRT object setup from which this
                                                          # calculate_opacity_contribution_spectra call is
                                                          # run, except for the fact that only one opacity
                                                          # species is used.
                                pressures=self._pressures * 1e-6,  # cgs to bar
                                wavelength_boundaries=self.wavelength_boundaries,
                                line_opacity_mode=self.line_opacity_mode,
                                line_by_line_opacity_sampling=self.line_by_line_opacity_sampling,
                                scattering_in_emission=self.scattering_in_emission,
                                emission_angle_grid=self.emission_angle_grid,
                                anisotropic_cloud_scattering=self.anisotropic_cloud_scattering,
                                path_input_data=self.path_input_data,
                                **fixed_opacity_sources, # use only the one species requested here.
                            )

                        if _default_mass_fractions is None: # Like above, will be initialized once,
                                                            # then reused.
                            _default_mass_fractions = {
                                species: jnp.zeros(
                                    self._pressures.size, dtype=FEAUTRIER_JAX_DTYPE
                                )
                            }

                        # Use the first line species in order to keep the resolving power consistent in all spectra
                        if opacity != "line_species": # This also means that calculate_opacity_contribution spectra
                                                      # will crash if there's no line_species.
                            fixed_opacity_sources["line_species"] = (
                                _default_radtrans.line_species # This only works since the first entry in
                                                               # "init" is "line_species", otherwise
                                                               # this would not even be filled at this point.
                            )

                        _radtrans = Radtrans( # This mirrors the pRT object setup from which this
                                              # calculate_opacity_contribution_spectra call is
                                              # run, except for the fact that only one opacity
                                              # species is used.
                            pressures=self._pressures * 1e-6,  # cgs to bar
                            wavelength_boundaries=self.wavelength_boundaries,
                            line_opacity_mode=self.line_opacity_mode,
                            line_by_line_opacity_sampling=self.line_by_line_opacity_sampling,
                            scattering_in_emission=self.scattering_in_emission,
                            emission_angle_grid=self.emission_angle_grid,
                            anisotropic_cloud_scattering=self.anisotropic_cloud_scattering,
                            path_input_data=self.path_input_data,
                            **fixed_opacity_sources,
                        )

                        # Generate the spectrum containing only the relevant fixed opacity source
                        spectral_parameters = {}

                        # Force all runtime opacity sources to their default (off) value, since only the
                        # current fixed opacity source should contribute to this spectrum. Note that these
                        # cannot be reached via kwargs below: they are named parameters of this function,
                        # so Python binds them directly instead of collecting them into **kwargs.
                        for _key, _value in opacity_source_default["runtime"].items():
                            if _key == "power_law":
                                # power_law's default is a nested dict of its own sub-parameters;
                                # unpack it into flat kwargs to match the spectral function signature
                                spectral_parameters["power_law_opacity_350nm"] = copy.deepcopy(
                                    _value["power_law_opacity_350nm"]
                                )
                                spectral_parameters["power_law_opacity_coefficient"] = copy.deepcopy(
                                    _value["power_law_opacity_coefficient"]
                                )
                            else:
                                spectral_parameters[_key] = copy.deepcopy(_value)

                        # Forward the remaining (non-opacity) spectral parameters as provided by the user
                        for _key, _value in kwargs.items():
                            if _key == "mass_fractions":
                                spectral_parameters[_key] = copy.deepcopy(_value)

                                # Fixes the first line species MMR to 0 so that it does not impact the spectrum
                                if species not in _default_mass_fractions:
                                    spectral_parameters[_key].update( # so if we are calling this on clouds
                                                                      # the first line species opacity won't
                                                                      # have any effect (updating a larger
                                                                      # dictionary with a smaller one
                                                                      # only updates those key values where
                                                                      # keys are the same (and adds newly
                                                                      # those entries that do not have a
                                                                      # corresponding key yet).
                                        _default_mass_fractions
                                    )
                            else:  # non-opacity spectral parameters
                                spectral_parameters[_key] = copy.deepcopy(_value)

                        # Update the user's cloud name to the full one everywhere it is needed to prevent crashes
                        if opacity == "cloud_species":
                            if _user_cloud_species != species:
                                for _key, _value in spectral_parameters.items():
                                    if isinstance(_value, dict):
                                        if _user_cloud_species in _value:
                                            _value[species] = copy.deepcopy(
                                                _value[_user_cloud_species]
                                            )
                                            del _value[_user_cloud_species]

                        opacity_contributions[opacity][species] = _radtrans.__getattribute__(
                            spectral_function
                        )(**spectral_parameters)

                        del _radtrans  # free up some memory
            else:  # variable (at runtime) opacity sources (also called analytical opacities)
                for opacity, opacity_value in opacity_source.items():
                    # opacity will be opaque_cloud_top_pressure, power_law, gray_opacity, ...
                    # And value will be its values
                    if opacity_value is None: # Skip if nothing is to be done.
                        opacity_contributions[opacity] = None
                        continue

                    spectral_parameters = {}

                    # Handle power law dual parameters special case
                    if opacity == "power_law":
                        for key, value in opacity_value.items():
                            spectral_parameters[key] = copy.deepcopy(value)

                        missing = [
                            key
                            for key, value in spectral_parameters.items()
                            if value is None
                        ]

                        if missing:
                            if len(missing) != len(spectral_parameters):
                                missing_names = "'" + "', '".join(missing) + "'"

                                debug.print(
                                    f"opacity '{opacity}' requires {len(spectral_parameters)} parameters, "
                                    f"but lacks value for parameters {missing_names}"
                                )

                            opacity_contributions[opacity] = None
                            continue
                    else:
                        spectral_parameters[opacity] = copy.deepcopy(opacity_value)

                    # Fill the spectral parameters with default values for non-relevant opacity sources
                    for key, value in kwargs.items():
                        if key == "mass_fractions":
                            spectral_parameters[key] = copy.deepcopy(
                                _default_mass_fractions
                            )
                        else:  # non-opacity spectral parameters
                            spectral_parameters[key] = copy.deepcopy(value)

                    opacity_contributions[opacity] = _default_radtrans.__getattribute__(
                        spectral_function
                    )(**spectral_parameters)

        return opacity_contributions

    @_compile_spectral_function(_FLUX_STATIC_ARGNAMES)
    def calculate_flux(
        self,
        temperatures: ArrayLike,
        mass_fractions: dict[str, ArrayLike],
        mean_molar_masses: ArrayLike,
        reference_gravity: float,
        planet_radius: float = None,
        opaque_cloud_top_pressure: float = None,
        cloud_particle_mean_radii: dict[str, ArrayLike] = None,
        cloud_particle_radius_distribution_std: float | dict[str, ArrayLike] = None,
        cloud_particle_radius_distribution: str = "lognormal",
        cloud_hansen_a: dict[str, ArrayLike] = None,
        cloud_hansen_b: dict[str, ArrayLike] = None,
        cloud_particle_number_density_grid: dict[str, ArrayLike] = None,
        clouds_particle_porosity_factor: dict[str, float] = None,
        cloud_f_sed: float | dict[str, float] = None,
        eddy_diffusion_coefficients: ArrayLike = None,
        haze_factor: float = 1.0,
        power_law_opacity_350nm: float = None,
        power_law_opacity_coefficient: float = None,
        gray_opacity: float = None,
        cloud_photosphere_wavelength_boundaries: ArrayLike = None,
        cloud_photosphere_median_optical_depth: float = None,
        cloud_fraction: float = 1.0,
        patchy_clouds: list[str] | tuple[str] = None,
        irradiation_geometry: str = "dayside_ave",
        stellar_intensities: ArrayLike = None,
        star_effective_temperature: float = None,
        star_radius: float = None,
        orbit_semi_major_axis: float = None,
        star_irradiation_angle: float = 0.0,
        reflectances: ArrayLike = None,
        emissivities: ArrayLike = None,
        additional_absorption_opacities_function: Callable = None,
        additional_scattering_opacities_function: Callable = None,
        adaptive_feautrier_iterations: bool = True,
        chunk_feautrier: bool = None,
        feautrier_chunk_size: int = None,
        thomas_min_batch: int = None,
        ck_chunk_size: int = None,
        frequencies_to_wavelengths: bool = True,
        return_contribution: bool = False,
        return_clear_spectrum: bool = False,
        return_photosphere_radius: bool = False,
        return_cloud_contribution: bool = False,
        return_opacities: bool = False,
        return_abundances: bool = False,
        return_stellar_intensities: bool = False,
        return_angle_dependent_intensities: bool = False,
        return_particle_radii_bins: bool = False,
        return_particle_radii: bool = False,
        target_particle_radii: ArrayLike = None,
        target_pressure: ArrayLike = None,
        pressures: ArrayLike = None,
        validate_inputs: bool = True, # Validate it used in the _compile_spectral_function
                                      # decorator, through a wrapper, so don't remove it here,
                                      # even if it is grayed out and appears to not be used.
        make_analytical_clouds_patchy=False,
        additional_opacity_functions_kwargs: dict = None,
    ) -> tuple[ArrayLike, ArrayLike, dict[str, Any]]:
        """Method to calculate the atmosphere's emitted flux (emission spectrum).

        Args:
            temperatures:
                The atmospheric temperature in K, at each atmospheric layer
                (1-d numpy array, same length as pressure array).
            mass_fractions:
                Dictionary of mass fractions for all atmospheric absorbers.
                Dictionary keys are the species names. Every mass fraction array has same length as pressure array.
            mean_molar_masses:
                The atmospheric mean molecular weight in amu, at each atmospheric layer
                (1-d numpy array, same length as pressure array).
            reference_gravity (float):
                Surface gravity in cgs. Vertically constant for emission spectra.
            planet_radius: planet radius at maximum pressure in cm. Only used to calculate the planet's changing
                Photospheric radius as function of wavelength, if return_photosphere_radius is True.
            opaque_cloud_top_pressure (Optional[float]):
                Pressure, in bar, where opaque cloud deck is added to the absorption opacity.
            cloud_particle_mean_radii (Optional):
                Dictionary of mean particle radii for all cloud species.
                Dictionary keys are the cloud species names. Every radius array has same length as pressure array.
            cloud_particle_radius_distribution_std:
                Width of the log-normal cloud particle size distribution.
                May be a shared scalar or a per-species dictionary whose
                values are scalars or layer profiles.
            cloud_particle_radius_distribution (Optional[string]):
                The cloud particle size distribution to use.
                Can be either 'lognormal' (default) or 'hansen'.
                If hansen, the cloud_hansen_b parameters must be used.
            cloud_hansen_a (Optional[dict]):
                A dictionary of the 'a' parameter values for each included cloud species and for each atmospheric
                layer, formatted as the kzz argument. Equivalent to cloud_particle_mean_radii.
                If cloud_hansen_a is not included and dist is "hansen", then it will be computed using Kzz and fsed
                (recommended).
            cloud_hansen_b (Optional[dict]):
                A dictionary of the 'b' parameter values for each included cloud species and for each atmospheric
                layer, formatted as the kzz argument. This is the width of the hansen distribution normalized by
                the particle area (1/cloud_hansen_a^2)
            cloud_particle_number_density_grid (Optional[dict]):
                EXPERIMENTAL
            clouds_particle_porosity_factor (Optional[dict]):
                A dictionary of porosity factors depending on the cloud species. This can be useful when opacities
                are calculated using the Distribution of Hollow Spheres (DHS) method.
            cloud_f_sed (Optional[float | dict[str, float]]):
                Cloud settling parameter, either use the same for all cloud species or hand over a dictionary
                that has one value per species, with the cloud species names as keys.
            eddy_diffusion_coefficients (Optional[float]):
                The atmospheric eddy diffusion coefficient in cgs (i.e. :math:`\\rm cm^2/s`), at each atmospheric
                layer (1-d numpy array, same length as pressure array).
            haze_factor (Optional[float]):
                Scalar factor, increasing the gas Rayleigh scattering cross-section.
            power_law_opacity_350nm (Optional[float]):
                Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
            power_law_opacity_coefficient (Optional[float]):
                Has to be given if kappa_zero is defined, this is the
                wavelength powerlaw index of the parametrized scattering
                opacity.
            gray_opacity (Optional[float]):
                Gray opacity value, to be added to the opacity at all pressures and wavelengths
                (units :math:`\\rm cm^2/g`)
            cloud_photosphere_median_optical_depth (Optional[float]):
                Median optical depth (across ``wavelength_boundaries``) of the clouds from the top of the
                atmosphere down to the gas-only photosphere. This parameter can be used for enforcing the presence
                of clouds in the photospheric region.
            cloud_photosphere_wavelength_boundaries (Optional[NDArray]):
                Min and max wavelength boundaries when calculating the median of the photospheric cloud optical
                depth. By default, the whole spectral range is used.
            cloud_fraction:
                Mix a column without cloud with a column with cloud to the requested fraction
                (0 = clear, 1 = full cloud cover). Cloud sources such as
                power-law clouds and opaque cloud decks follow the same mixing.
                By default, assume a full cloud cover.
            patchy_clouds:
                List of cloud species affected by ``cloud_fraction``.
            irradiation_geometry (Optional[string]):
                if equal to ``'dayside_ave'``: use the dayside average geometry.
                If equal to ``'planetary_ave'``: use the planetary average geometry.
                If equal to ``'non-isotropic'``: use the non-isotropic geometry.
            stellar_intensities (Optional[array]):
                The stellar intensity to use. If None, it will be calculated using a PHOENIX model.
            star_effective_temperature (Optional[float]):
                The temperature of the host star in K, used only if the
                scattering is considered. If not specified, the direct light contribution is not calculated.
            star_radius (Optional[float]):
                The radius of the star in cm. If specified, used to scale the to scale the stellar flux,
                otherwise it uses PHOENIX radius.
            orbit_semi_major_axis (Optional[float]):
                The distance of the planet from the star. Used to scale the stellar flux when the scattering of the
                direct light is considered.
            star_irradiation_angle (Optional[float]):
                Inclination angle of the direct light with respect to the normal to the atmosphere. Used only in
                the non-isotropic geometry scenario.
            reflectances (Optional):
                Reflectances of the surface (layer with the highest pressure).
            emissivities (Optional):
                Emissivities of the surface (layer with the highest pressure).
            additional_absorption_opacities_function (Optional[function]):
                A python function that takes wavelength arrays in microns and pressure arrays in bars
                as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                number of wavelength points x number of pressure points.
                This opacity will then be added to the atmospheric absorption opacity.
                This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                because line opacities require a proper correlated-k treatment.
                It may be used to add simple cloud absorption laws, for example, which
                have opacities that vary only slowly with wavelength, such that the current
                model resolution is sufficient to resolve any variations.
                This is a static (compile-time) argument: pass a stable function reference.
                Passing a freshly created lambda or closure (e.g., the product of a function factory)
                on each call forces a full recompilation of the spectral kernel.
                Relatedly, make sure that your function is jax.jit compilable.
            additional_scattering_opacities_function (Optional[function]):
                A python function that takes wavelength arrays in microns and pressure arrays in bars
                as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                number of wavelength points x number of pressure points.
                This opacity will then be added to the atmospheric absorption opacity.
                It may be used to add simple cloud absorption laws, for example, which
                have opacities that vary only slowly with wavelength, such that the current
                model resolution is sufficient to resolve any variations.
                This is a static (compile-time) argument: pass a stable function reference.
                Passing a freshly created lambda or closure (e.g., the product of a function factory)
                on each call forces a full recompilation of the spectral kernel.
                Relatedly, make sure that your function is jax.jit compilable.
            additional_opacity_functions_kwargs (Optional[dict]):
                Dictionary of extra keyword arguments that is unpacked into both
                additional_absorption_opacities_function and additional_scattering_opacities_function,
                on top of the wavelengths and pressures those always receive. If you supply both
                functions, each of them must be able to accept every key in this dictionary.
                In contrast to the functions themselves, the values in this dictionary are traced:
                use it to pass the parameters of your opacity model, so that changing them neither
                triggers a recompilation nor breaks differentiability.
            adaptive_feautrier_iterations (Optional[bool]):
                If True, adapt the number of iterations of the Feautrier scattering-method
                (scattering_in_emission=True) based on the photon destruction probability.
            chunk_feautrier (Optional[bool]):
                If True, evaluate the Feautrier solver in frequency chunks. If None (default),
                resolves to True.
            feautrier_chunk_size (Optional[int]):
                Number of frequency bins per chunk when ``chunk_feautrier`` is enabled. If None
                (default), resolves to 1536. Both chunking settings are static (compile-time)
                arguments: every distinct (chunk_feautrier, feautrier_chunk_size) pair compiles
                its own kernel.
            thomas_min_batch (Optional[int]):
                Batch size (``n_gf * n_angles``) above which the Thomas-factorisation Feautrier
                solver path is used over PCR on GPU. If None (default), resolves to
                ``linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH``. Static (compile-time) argument.
            ck_chunk_size (Optional[int]):
                Number of (frequency, layer) cells per chunk in the pure-JAX correlated-k
                mixing kernel, used to cap peak memory on memory-constrained runs. If None
                (default), resolves to ``radiative_transfer_core.DEFAULT_CK_CHUNK_SIZE`` (0,
                chunking disabled). Static (compile-time) argument.
            frequencies_to_wavelengths (Optional[bool]):
                if True, convert the frequencies (Hz) output to wavelengths (cm),
                and the flux per frequency output (erg.s-1.cm-2/Hz) to flux per wavelength (erg.s-1.cm-2/cm)
            return_contribution (Optional[bool]):
                If True, calculate the emission contribution function.
            return_clear_spectrum (Optional[bool]):
                If True, return the clear spectrum in addition to a cloudy spectrum.
            return_photosphere_radius (Optional[bool]):
                If True, calculate and return the photosphere radius.
                If return clear spectrum is True, the code will calculate the photosphere radius also for the
                clear flux branch, and return it under "clear_photosphere_radius" in addition to "photosphere_radius".
            return_cloud_contribution (Optional[bool]):
                If True, calculate the cloud contribution.
            return_opacities (Optional[bool]):
                If True, return the absorption opacities and scattering opacities for species and clouds, as well
                as optical depths
            return_abundances (Optional[bool]):
                If True, return the mass fractions.
            return_stellar_intensities (Optional[bool]):
                If True, include the stellar intensities used by the emission
                solver in the auxiliary output mapping.
            return_angle_dependent_intensities (Optional[bool]):
                If True, include the angle-dependent intensities in the
                auxiliary output mapping.
            return_particle_radii_bins (Optional[bool]):
                If True, return the particle radii bins.
            return_particle_radii (Optional[bool]):
                If True, return the particle radii.
            target_particle_radii (Optional[ArrayLike]:
                Particle radius to target for selective cloud bin disabling. Radius bin closest
                the input value will be disabled.
            target_pressure (Optional[ArrayLike]:
                Pressure to target for selective cloud bin disabling. Pressure bin closest to
                the input value will be disabled.
            validate_inputs (Optional[bool]):
                If True (default), check the reference gravity is positive. Under
                ``jax.jit`` this check needs a device-to-host callback on every
                evaluation, which is costly for GPU performance. Set to False
                for retrievals or if you're sure inputs have already been checked.
            pressures (Optional[ArrayLike]):
                If not None, these pressures will be used as the vertical structure
                describing the atmosphere. Pressures must be in units of bar and
                must be sorted in increasing order. They will be converted to cgs
                units inside pRT.
            make_analytical_clouds_patchy (Optional[bool]):
                False by default. If True, analytical clouds (opaque cloud top pressure,
                power law opacity) will be treated as patchy clouds.
        """

        # The Feautrier computation is "chunked", so broken into frequency
        # parts for memory efficiency and to allow for more efficient XLA-threading.
        chunk_feautrier, feautrier_chunk_size = (
            radiative_transfer_core.resolve_feautrier_chunking(
                chunk_feautrier=chunk_feautrier,
                feautrier_chunk_size=feautrier_chunk_size,
            )
        )
        # See docstring for meaning.
        if thomas_min_batch is None:
            thomas_min_batch = linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH
        # See docstring for meaning.
        if ck_chunk_size is None:
            ck_chunk_size = radiative_transfer_core.DEFAULT_CK_CHUNK_SIZE

        # Check whether user-specified pressure array is tp be used
        solver_pressures = self._resolve_solver_pressures(pressures)

        (
            anisotropic_cloud_scattering,
            patchy_clouds,
            sum_opacities,
        ) = self._init_spectral_function(
            is_emission=True,
            patchy_clouds=patchy_clouds,
        )

        return self._calculate_flux_kernel(
            pressures=solver_pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=_convert_scalar_to_strong_type(reference_gravity),
            planet_radius=_convert_scalar_to_strong_type(planet_radius),
            opaque_cloud_top_pressure=_convert_scalar_to_strong_type(opaque_cloud_top_pressure),
            cloud_particle_mean_radii=cloud_particle_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            haze_factor=_convert_scalar_to_strong_type(haze_factor),
            power_law_opacity_350nm=_convert_scalar_to_strong_type(power_law_opacity_350nm),
            power_law_opacity_coefficient=_convert_scalar_to_strong_type(power_law_opacity_coefficient),
            gray_opacity=_convert_scalar_to_strong_type(gray_opacity),
            cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
            cloud_photosphere_median_optical_depth=_convert_scalar_to_strong_type(cloud_photosphere_median_optical_depth),
            cloud_fraction=_convert_scalar_to_strong_type(cloud_fraction),
            patchy_clouds=patchy_clouds,
            irradiation_geometry=irradiation_geometry,
            stellar_intensities=stellar_intensities,
            star_effective_temperature=_convert_scalar_to_strong_type(star_effective_temperature),
            star_radius=_convert_scalar_to_strong_type(star_radius),
            orbit_semi_major_axis=_convert_scalar_to_strong_type(orbit_semi_major_axis),
            star_irradiation_angle=_convert_scalar_to_strong_type(star_irradiation_angle),
            reflectances=reflectances,
            emissivities=emissivities,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            adaptive_feautrier_iterations=adaptive_feautrier_iterations,
            chunk_feautrier=chunk_feautrier,
            feautrier_chunk_size=feautrier_chunk_size,
            thomas_min_batch=thomas_min_batch,
            ck_chunk_size=ck_chunk_size,
            frequencies_to_wavelengths=frequencies_to_wavelengths,
            return_contribution=return_contribution,
            return_clear_spectrum=return_clear_spectrum,
            return_photosphere_radius=return_photosphere_radius,
            return_cloud_contribution=return_cloud_contribution,
            return_opacities=return_opacities,
            return_abundances=return_abundances,
            return_stellar_intensities=return_stellar_intensities,
            return_angle_dependent_intensities=return_angle_dependent_intensities,
            return_particle_radii_bins=return_particle_radii_bins,
            return_particle_radii=return_particle_radii,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
            sum_opacities=sum_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            make_analytical_clouds_patchy=make_analytical_clouds_patchy,
            additional_opacity_functions_kwargs=additional_opacity_functions_kwargs,
        )

    @jit(
        static_argnames=[
            "cloud_particle_radius_distribution",
            "adaptive_feautrier_iterations",
            "irradiation_geometry",
            "chunk_feautrier",
            "feautrier_chunk_size",
            "thomas_min_batch",
            "ck_chunk_size",
            "frequencies_to_wavelengths",
            "patchy_clouds",
            "sum_opacities",
            "anisotropic_cloud_scattering",
            "additional_absorption_opacities_function",
            "additional_scattering_opacities_function",
            "return_contribution",
            "return_clear_spectrum",
            "return_photosphere_radius",
            "return_cloud_contribution",
            "return_opacities",
            "return_abundances",
            "return_stellar_intensities",
            "return_angle_dependent_intensities",
            "return_particle_radii_bins",
            "return_particle_radii",
            "make_analytical_clouds_patchy",
        ]
    )
    def _calculate_flux_kernel(
        self,
        temperatures,
        mass_fractions,
        mean_molar_masses,
        reference_gravity,
        pressures=None,
        planet_radius=None,
        opaque_cloud_top_pressure=None,
        cloud_particle_mean_radii=None,
        cloud_particle_radius_distribution_std=None,
        cloud_particle_radius_distribution="lognormal",
        cloud_hansen_a=None,
        cloud_hansen_b=None,
        cloud_particle_number_density_grid=None,
        clouds_particle_porosity_factor=None,
        cloud_f_sed=None,
        eddy_diffusion_coefficients=None,
        haze_factor=1.0,
        power_law_opacity_350nm=None,
        power_law_opacity_coefficient=None,
        gray_opacity=None,
        cloud_photosphere_wavelength_boundaries=None,
        cloud_photosphere_median_optical_depth=None,
        cloud_fraction=1.0,
        patchy_clouds=None,
        irradiation_geometry="dayside_ave",
        stellar_intensities=None,
        star_effective_temperature=None,
        star_radius=None,
        orbit_semi_major_axis=None,
        star_irradiation_angle=0.0,
        reflectances=None,
        emissivities=None,
        additional_absorption_opacities_function=None,
        additional_scattering_opacities_function=None,
        adaptive_feautrier_iterations=True,
        chunk_feautrier=True,
        feautrier_chunk_size=1536,
        thomas_min_batch=linalg.DEFAULT_FEAUTRIER_THOMAS_MIN_BATCH,
        ck_chunk_size=radiative_transfer_core.DEFAULT_CK_CHUNK_SIZE,
        frequencies_to_wavelengths=True,
        return_contribution=False,
        return_clear_spectrum=False,
        return_photosphere_radius=False,
        return_cloud_contribution=False,
        return_opacities=False,
        return_abundances=False,
        return_stellar_intensities=False,
        return_angle_dependent_intensities=False,
        return_particle_radii_bins=False,
        return_particle_radii=False,
        target_particle_radii=None,
        target_pressure=None,
        sum_opacities=True,
        anisotropic_cloud_scattering=None,
        make_analytical_clouds_patchy=False,
        additional_opacity_functions_kwargs=None,
    ):

        """JIT-compiled kernel for calculate_flux. See calculate_flux for parameter docs."""
        # This is star_irradiation_cos_angle, stellar_intensities, reflectances, emissivities
        # (normalized)
        emission_inputs = _sanitize_emission_surface_inputs(
            frequencies=self._frequencies,
            scattering_in_emission=self._scattering_in_emission,
            star_irradiation_angle=star_irradiation_angle,
            stellar_intensities=stellar_intensities,
            star_effective_temperature=star_effective_temperature,
            star_radius=star_radius,
            orbit_semi_major_axis=orbit_semi_major_axis,
            reflectances=reflectances,
            emissivities=emissivities,
        )
        sanitized_inputs = _sanitize_flux_opacity_inputs(
            patchy_clouds=patchy_clouds,
            cloud_particle_mean_radii=cloud_particle_mean_radii,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
            cloud_species=self._cloud_species,
            missing_float=_MISSING_FLOAT,
            missing_neg_inf=_MISSING_NEG_INF,
            additional_opacities_placeholder_fn=_additional_opacities_placeholder_fn,
        )
        partial_coverage_cloud_species = self._get_partial_coverage_cloud_species(
            sanitized_inputs["patchy_clouds"]
        )
        has_partial_coverage_analytical_cloud = (
            self._has_partial_coverage_analytical_cloud(
                opaque_cloud_top_pressure=opaque_cloud_top_pressure,
                power_law_opacity_350nm=power_law_opacity_350nm,
                power_law_opacity_coefficient=power_law_opacity_coefficient,
                make_analytical_clouds_patchy=make_analytical_clouds_patchy,
            )
        )

        # Determine whether clear opacities are needed.
        if (
            len(partial_coverage_cloud_species) > 0
            or has_partial_coverage_analytical_cloud
        ):
            should_compute_opacities_without_patchy_clouds = True
        else:
            should_compute_opacities_without_patchy_clouds = False

        (
            opacities,
            continuum_opacities_scattering,
            cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities,
            cloud_opacities,
            cloud_particle_mean_radii,
            cloud_scattering_in_transmission,
            clear_opacities, # todo better name here based on discussion
            clear_continuum_opacities_scattering,
            clear_cloud_anisotropic_scattering_opacities,
            clear_cloud_absorption_opacities,
        ) = self._calculate_opacities(
            pressures=pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            opaque_cloud_top_pressure=sanitized_inputs["opaque_cloud_top_pressure"],
            cloud_particle_mean_radii=sanitized_inputs["cloud_particle_mean_radii"],
            cloud_particle_radius_distribution_std=sanitized_inputs[
                "cloud_particle_radius_distribution_std"
            ],
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_hansen_a=sanitized_inputs["cloud_hansen_a"],
            cloud_hansen_b=sanitized_inputs["cloud_hansen_b"],
            cloud_particle_number_density_grid=sanitized_inputs[
                "cloud_particle_number_density_grid"
            ],
            clouds_particle_porosity_factor=sanitized_inputs[
                "clouds_particle_porosity_factor"
            ],
            cloud_f_sed=sanitized_inputs["cloud_f_sed"],
            eddy_diffusion_coefficients=sanitized_inputs["eddy_diffusion_coefficients"],
            haze_factor=haze_factor,
            power_law_opacity_350nm=sanitized_inputs["power_law_opacity_350nm"],
            power_law_opacity_coefficient=sanitized_inputs[
                "power_law_opacity_coefficient"
            ],
            gray_opacity=sanitized_inputs["gray_opacity"],
            cloud_photosphere_median_optical_depth=sanitized_inputs[
                "cloud_photosphere_median_optical_depth"
            ],
            patchy_clouds=sanitized_inputs["patchy_clouds"],
            return_cloud_contribution=return_cloud_contribution,
            additional_absorption_opacities_function=sanitized_inputs[
                "additional_absorption_opacities_function"
            ],
            additional_scattering_opacities_function=sanitized_inputs[
                "additional_scattering_opacities_function"
            ],
            compute_clear_opacities=should_compute_opacities_without_patchy_clouds,
            return_opacities=return_opacities,
            sum_opacities=sum_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            target_particle_radii=sanitized_inputs["target_particle_radii"],
            target_pressure=sanitized_inputs["target_pressure"],
            ck_chunk_size=ck_chunk_size,
            make_analytical_clouds_patchy=make_analytical_clouds_patchy,
            additional_opacity_functions_kwargs=additional_opacity_functions_kwargs,
        )

        (
            flux,
            emission_contribution,
            angle_dependent_intensities,
            optical_depths,
            relative_cloud_scaling_factor,
        ) = self._calculate_flux(
            pressures=pressures,
            temperatures=temperatures,
            reference_gravity=reference_gravity,
            opacities=opacities,
            continuum_opacities_scattering=continuum_opacities_scattering,
            irradiation_geometry=irradiation_geometry,
            star_irradiation_cos_angle=emission_inputs["star_irradiation_cos_angle"],
            stellar_intensity=emission_inputs["stellar_intensities"],
            reflectances=emission_inputs["reflectances"],
            emissivities=emission_inputs["emissivities"],
            return_contribution=return_contribution,
            return_angle_dependent_intensities=return_angle_dependent_intensities,
            cloud_f_sed=cloud_f_sed,
            photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
            cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
            cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities=cloud_absorption_opacities,
            adaptive_feautrier_iterations=adaptive_feautrier_iterations,
            sum_opacities=sum_opacities,
            chunk_feautrier=chunk_feautrier,
            feautrier_chunk_size=feautrier_chunk_size,
            thomas_min_batch=thomas_min_batch,
        )

        # --- Cloud blending for partial coverage ---
        flux_clear = flux  # default: clear flux just means all patchy clouds are turned off.
                           # so a clear flux may be affected by clouds.
                           # what is called flux_clear is still the nominal patchy flux,
                           # however, it will be overwritten below.

        if return_clear_spectrum:
            # Always compute clear flux when the caller requested it.
            flux_clear, _, _, clear_optical_depths, _ = self._calculate_flux( # TODO return "less cloudy spectrum"
                pressures=pressures,
                temperatures=temperatures,
                reference_gravity=reference_gravity,
                opacities=clear_opacities,
                continuum_opacities_scattering=clear_continuum_opacities_scattering,
                irradiation_geometry=irradiation_geometry,
                star_irradiation_cos_angle=emission_inputs[
                    "star_irradiation_cos_angle"
                ],
                stellar_intensity=emission_inputs["stellar_intensities"],
                reflectances=emission_inputs["reflectances"],
                emissivities=emission_inputs["emissivities"],
                return_contribution=False,
                cloud_f_sed=cloud_f_sed,
                photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                cloud_anisotropic_scattering_opacities=clear_cloud_anisotropic_scattering_opacities,
                cloud_absorption_opacities=clear_cloud_absorption_opacities,
                adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                sum_opacities=sum_opacities,
                chunk_feautrier=chunk_feautrier,
                feautrier_chunk_size=feautrier_chunk_size,
                thomas_min_batch=thomas_min_batch,
            )

            def _blend_flux(args):
                flux, flux_clear, cloud_fraction = args
                # Cast back to the cloudy-flux dtype: a cloud_fraction of a
                # wider dtype would otherwise promote this branch only, and
                # lax.cond requires both branches to agree.
                return (
                    cloud_fraction * flux + (1 - cloud_fraction) * flux_clear
                ).astype(flux.dtype)

            def _pass_through_flux(args):
                return args[0]

            flux = lax.cond(
                jnp.less(cloud_fraction, 1),

                _blend_flux,
                _pass_through_flux,
                (flux, flux_clear, cloud_fraction),
            )
        elif should_compute_opacities_without_patchy_clouds:
            # Only compute clear flux and blend when cloud_fraction < 1.
            def _blend_partial_cloud_flux(cloudy_flux):
                clear_flux, _, _, _, _ = self._calculate_flux(
                    pressures=pressures,
                    temperatures=temperatures,
                    reference_gravity=reference_gravity,
                    opacities=clear_opacities,
                    continuum_opacities_scattering=clear_continuum_opacities_scattering,
                    irradiation_geometry=irradiation_geometry,
                    star_irradiation_cos_angle=emission_inputs[
                        "star_irradiation_cos_angle"
                    ],
                    stellar_intensity=emission_inputs["stellar_intensities"],
                    reflectances=emission_inputs["reflectances"],
                    emissivities=emission_inputs["emissivities"],
                    return_contribution=False,
                    cloud_f_sed=cloud_f_sed,
                    photospheric_cloud_optical_depths=cloud_photosphere_median_optical_depth,
                    cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                    cloud_anisotropic_scattering_opacities=clear_cloud_anisotropic_scattering_opacities,
                    cloud_absorption_opacities=clear_cloud_absorption_opacities,
                    adaptive_feautrier_iterations=adaptive_feautrier_iterations,
                    sum_opacities=sum_opacities,
                    chunk_feautrier=chunk_feautrier,
                    feautrier_chunk_size=feautrier_chunk_size,
                    thomas_min_batch=thomas_min_batch,
                )

                return (
                    cloud_fraction * cloudy_flux + (1 - cloud_fraction) * clear_flux
                ).astype(cloudy_flux.dtype)

            def _pass_through_cloudy_flux(flux):
                return flux

            flux = lax.cond(
                jnp.less(cloud_fraction, 1),
                _blend_partial_cloud_flux,
                _pass_through_cloudy_flux,
                flux,
            )

        # --- Build additional outputs ---
        additional_outputs = {}

        if (
            sanitized_inputs["cloud_particle_mean_radii"] is not None
            or len(sanitized_inputs["cloud_f_sed"]) > 0
        ):
            additional_outputs["cloud_particle_mean_radii"] = cloud_particle_mean_radii

        if return_particle_radii_bins:
            additional_outputs["particle_radii_bins"] = self._clouds_loaded_opacities[
                "particle_radii_bins"
            ]

        if return_particle_radii:
            additional_outputs["particle_radii"] = self._clouds_loaded_opacities[
                "particle_radii"
            ]

        if return_stellar_intensities:
            additional_outputs["stellar_intensities"] = emission_inputs[
                "stellar_intensities"
            ]

        if return_contribution:
            additional_outputs["emission_contribution"] = emission_contribution

        if return_clear_spectrum:
            if frequencies_to_wavelengths:
                additional_outputs["clear_spectrum"] = flux_hz2flux_cm(
                    flux_hz=flux_clear, frequency=self._frequencies
                )
            else:
                additional_outputs["clear_spectrum"] = flux_clear

        if return_photosphere_radius:
            if planet_radius is not None:
                additional_outputs["photosphere_radius"] = (
                    self._calculate_photosphere_radius(
                        pressures=pressures * 1e-6,
                        temperatures=temperatures,
                        mean_molar_masses=mean_molar_masses,
                        reference_gravity=reference_gravity,
                        planet_radius=planet_radius,
                        opacities=opacities,
                        continuum_opacities_scattering=continuum_opacities_scattering,
                        cloud_f_sed=cloud_f_sed,
                        cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
                        cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                        cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
                        cloud_absorption_opacities=cloud_absorption_opacities,
                        optical_depths=optical_depths,
                    )
                )
                if return_clear_spectrum:
                    additional_outputs["clear_photosphere_radius"] = (
                        self._calculate_photosphere_radius(
                            pressures=pressures * 1e-6,
                            temperatures=temperatures,
                            mean_molar_masses=mean_molar_masses,
                            reference_gravity=reference_gravity,
                            planet_radius=planet_radius,
                            opacities=clear_opacities,
                            continuum_opacities_scattering=clear_continuum_opacities_scattering,
                            cloud_f_sed=cloud_f_sed,
                            cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
                            cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                            cloud_anisotropic_scattering_opacities=clear_cloud_anisotropic_scattering_opacities,
                            cloud_absorption_opacities=clear_cloud_absorption_opacities,
                            optical_depths=clear_optical_depths,
                        )
                    )
            else:
                additional_outputs["photosphere_radius"] = jnp.zeros(
                    self._frequencies.size, dtype=FEAUTRIER_JAX_DTYPE
                )

        if return_cloud_contribution and len(self._cloud_species) > 0:
            additional_outputs["cloud_contribution"] = (
                clouds.compute_species_optical_depths( # TODO Annie, since cloud related.
                                                       #    remove when done.
                    reference_gravity=reference_gravity,
                    pressures=pressures,
                    cloud_opacities=cloud_opacities,
                )
            )
        if return_angle_dependent_intensities:
            additional_outputs["angle_dependent_intensities"] = (
                angle_dependent_intensities
            )
        if return_opacities:
            additional_outputs["opacities"] = opacities
            additional_outputs["continuum_opacities_scattering"] = (
                continuum_opacities_scattering
            )
            additional_outputs["cloud_absorption_opacities"] = (
                cloud_absorption_opacities
            )
            additional_outputs["cloud_anisotropic_scattering_opacities"] = (
                cloud_anisotropic_scattering_opacities
            )
            additional_outputs["optical_depths"] = optical_depths

        if return_abundances:
            additional_outputs["abundances"] = mass_fractions

        if relative_cloud_scaling_factor is not None:
            additional_outputs["relative_cloud_scaling_factor"] = (
                relative_cloud_scaling_factor
            )

        if frequencies_to_wavelengths:
            return (
                frequency2wavelength(self._frequencies),
                flux_hz2flux_cm(flux_hz=flux, frequency=self._frequencies),
                additional_outputs,
            )
        else:
            return self._frequencies, flux, additional_outputs

    def _calculate_photosphere_radius(
        self,
        temperatures: ArrayLike,
        mean_molar_masses: ArrayLike,
        reference_gravity: float,
        planet_radius: float,
        opacities: ArrayLike,
        continuum_opacities_scattering: ArrayLike,
        cloud_f_sed: float,
        cloud_photosphere_median_optical_depth: float,
        cloud_anisotropic_scattering_opacities: ArrayLike,
        cloud_absorption_opacities: ArrayLike,
        optical_depths: ArrayLike = None,
        cloud_photosphere_wavelength_boundaries: ArrayLike = None,
        pressures: ArrayLike = None,
    ) -> ArrayLike:
        """Estimate the wavelength-dependent photosphere radius.

        The photosphere radius is obtained by computing or reusing optical
        depths, locating the layer where the angle-averaged optical depth
        reaches $2/3$, and interpolating the hydrostatic radius profile to that
        pressure.

        Args:
            temperatures:
                Atmospheric temperature profile in K.
            mean_molar_masses:
                Mean molecular mass profile in amu.
            reference_gravity:
                Reference gravity in cgs units.
            planet_radius:
                Radius in cm at the deepest atmospheric layer.
            opacities:
                Total opacity array used for the emission calculation.
            continuum_opacities_scattering:
                Continuum scattering opacity array.
            cloud_f_sed:
                Cloud settling parameter used when photospheric cloud scaling is
                active.
            cloud_photosphere_median_optical_depth:
                Optional target cloud optical depth near the photosphere.
            cloud_anisotropic_scattering_opacities:
                Cloud anisotropic scattering opacities.
            cloud_absorption_opacities:
                Cloud absorption opacities.
            optical_depths:
                Precomputed optical depths. If omitted, they are recomputed.
            cloud_photosphere_wavelength_boundaries:
                Optional wavelength bounds used for the photospheric cloud model.
            pressures (Oprional[ArrayLike]):
                If not None, these pressures will be used as the vertical structure
                describing the atmosphere. Pressures must be in units of bar and
                must be sorted in increasing order. They will be converted to cgs
                units inside pRT.

        Returns:
            Photosphere radius on the internal frequency grid, in cm.
        """
        sum_opacities = self._set_sum_opacities(
            emission=True,
            line_opacity_mode=self._line_opacity_mode,
            scattering_in_emission=self._scattering_in_emission,
        )

        # Check whether user-specified pressure array is to be used
        solver_pressures = self._resolve_solver_pressures(pressures)

        if planet_radius is not None:
            radius_hydrostatic_equilibrium = (
                self._calculate_radius_hydrostatic_equilibrium(
                    pressures=solver_pressures, # pressures[-1] is the reference pressure
                                                # because within
                                                # _calculate_radius_hydrostatic_equilibrium()
                                                # compute_radius_hydrostatic_equilibrium()
                                                # is called with
                                                # reference_pressure=pressures[-1]
                    temperatures=temperatures,
                    mean_molar_masses=mean_molar_masses,
                    reference_gravity=reference_gravity,
                    planet_radius=planet_radius, # this is the reference radius
                )
            )

            # In the current call pattern where _calculate_photosphere_radius is only
            # called from within calculate_flux, nothing happens here, since
            # optical_depths is already handed over from calculate_flux().
            optical_depths = self._get_photosphere_optical_depths(
                pressures=solver_pressures,
                optical_depths=optical_depths,
                reference_gravity=reference_gravity,
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                cloud_f_sed=cloud_f_sed,
                cloud_photosphere_median_optical_depth=cloud_photosphere_median_optical_depth,
                cloud_photosphere_wavelength_boundaries=cloud_photosphere_wavelength_boundaries,
                cloud_anisotropic_scattering_opacities=cloud_anisotropic_scattering_opacities,
                cloud_absorption_opacities=cloud_absorption_opacities,
                sum_opacities=sum_opacities,
            )

            return self._interpolate_photosphere_radius_profile(
                pressures=solver_pressures,
                optical_depths=optical_depths,
                radius_hydrostatic_equilibrium=radius_hydrostatic_equilibrium,
            )

        return jnp.zeros(self._frequencies.size, dtype=FEAUTRIER_JAX_DTYPE)

    def calculate_rosseland_planck_opacities(
        self,
        temperatures: npt.NDArray[float],
        mass_fractions: dict[str, npt.NDArray[float]],
        mean_molar_masses: npt.NDArray[float],
        reference_gravity: float,
        opaque_cloud_top_pressure: float = None,
        cloud_particle_mean_radii: dict[str, ArrayLike] = None,
        cloud_particle_radius_distribution_std: float | dict[str, ArrayLike] = None,
        cloud_particle_radius_distribution: str = "lognormal",
        cloud_hansen_a: dict[str, ArrayLike] = None,
        cloud_hansen_b: dict[str, ArrayLike] = None,
        clouds_particle_porosity_factor: dict[str, float] = None,
        cloud_f_sed: float = None,
        eddy_diffusion_coefficients: float = None,
        haze_factor: float = 1.0,
        power_law_opacity_350nm: float = None,
        power_law_opacity_coefficient: float = None,
        gray_opacity: float = None,
        additional_absorption_opacities_function: Callable = None,
        additional_scattering_opacities_function: Callable = None,
        additional_opacity_functions_kwargs: dict = None,
    ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Method to calculate the atmosphere's Rosseland and Planck mean opacities.

        Args:
            temperatures:
                the atmospheric temperature in K, at each atmospheric layer
                (1-d numpy array, same length as pressure array).
            mass_fractions:
                dictionary of mass fractions for all atmospheric absorbers.
                Dictionary keys are the species names.
                Every mass fraction array
                has same length as pressure array.
            mean_molar_masses:
                the atmospheric mean molecular weight in amu,
                at each atmospheric layer
                (1-d numpy array, same length as pressure array).
            reference_gravity (float):
                Surface gravity in cgs. Vertically constant for emission
                spectra.
            opaque_cloud_top_pressure (Optional[float]):
                Pressure, in bar, where opaque cloud deck is added to the
                absorption opacity.
            cloud_particle_mean_radii (Optional):
                dictionary of mean particle radii for all cloud species.
                Dictionary keys are the cloud species names.
                Every radius array has same length as pressure array.
            cloud_particle_radius_distribution_std:
                Width of the log-normal cloud particle size distribution.
                May be a shared scalar or a per-species dictionary whose
                values are scalars or layer profiles.
            cloud_particle_radius_distribution (Optional[string]):
                The cloud particle size distribution to use.
                Can be either 'lognormal' (default) or 'hansen'.
                If hansen, the cloud_hansen_b parameters must be used.
            cloud_hansen_a (Optional[dict]):
                A dictionary of the 'a' parameter values for each
                included cloud species and for each atmospheric layer,
                formatted as the kzz argument. Equivalent to radius arg.
                If cloud_hansen_a is not included and dist is "hansen", then it will
                be computed using Kzz and fsed (recommended).
            cloud_hansen_b (Optional[dict]):
                A dictionary of the 'b' parameter values for each
                included cloud species and for each atmospheric layer,
                formatted as the kzz argument. This is the width of the hansen
                distribution normalized by the particle area (1/cloud_hansen_a^2)
            clouds_particle_porosity_factor (Optional[dict]):
                A dictionary containing the particle porosity factor for each cloud species.
            cloud_f_sed (Optional[float]):
                cloud settling parameter
            eddy_diffusion_coefficients (Optional):
                the atmospheric eddy diffusion coefficient in cgs
                (i.e. :math:`\\rm cm^2/s`),
                at each atmospheric layer
                (1-d numpy array, same length as pressure array).
            haze_factor (Optional[float]):
                Scalar factor, increasing the gas Rayleigh scattering
                cross-section.
            power_law_opacity_350nm (Optional[float]):
                Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
            power_law_opacity_coefficient (Optional[float]):
                Has to be given if kappa_zero is defined, this is the
                wavelength powerlaw index of the parametrized scattering
                opacity.
            gray_opacity (Optional[float]):
                Gray opacity value, to be added to the opacity at all
                pressures and wavelengths (units :math:`\\rm cm^2/g`)
            TODO: add support for additional_absorption_opacities_function,
                additional_scattering_opacities_function and
                cloud_photosphere_median_optical_depth.
        """
        if not self._scattering_in_emission:
            raise ValueError(
                "pRT must run in scattering_in_emission = True mode to calculate kappa_Rosseland and kappa_Planck'"
            )

        auto_anisotropic_cloud_scattering = False

        if self._anisotropic_cloud_scattering == "auto":
            self._anisotropic_cloud_scattering = True
            auto_anisotropic_cloud_scattering = True
        elif not self._anisotropic_cloud_scattering:
            debug.print(
                f"anisotropic cloud scattering is recommended when calculating rosseland and planck"
                f"mean opacities to be used in the emission context (as opposed to transmission geometry), "
                f"but 'anisotropic_cloud_scattering' was set to {self._anisotropic_cloud_scattering}; "
                f"set it to 'auto' to disable this warning."
            )

        sanitized_inputs = _sanitize_flux_opacity_inputs(
            patchy_clouds=(),
            cloud_particle_mean_radii=cloud_particle_mean_radii,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_median_optical_depth=None,
            cloud_particle_number_density_grid=None,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            target_particle_radii=None,
            target_pressure=None,
            cloud_species=self._cloud_species,
            missing_float=_MISSING_FLOAT,
            missing_neg_inf=_MISSING_NEG_INF,
            additional_opacities_placeholder_fn=_additional_opacities_placeholder_fn,
        )

        # No photospheric cloud in Rosseland or Planck opacities, but could be added.
        (
            opacities,
            continuum_opacities_scattering,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self._calculate_opacities(
            pressures=self._pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            opaque_cloud_top_pressure=sanitized_inputs["opaque_cloud_top_pressure"],
            cloud_particle_mean_radii=sanitized_inputs["cloud_particle_mean_radii"],
            cloud_particle_radius_distribution_std=sanitized_inputs[
                "cloud_particle_radius_distribution_std"
            ],
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_hansen_a=sanitized_inputs["cloud_hansen_a"],
            cloud_hansen_b=sanitized_inputs["cloud_hansen_b"],
            cloud_particle_number_density_grid=sanitized_inputs[
                "cloud_particle_number_density_grid"
            ],
            clouds_particle_porosity_factor=sanitized_inputs[
                "clouds_particle_porosity_factor"
            ],
            cloud_f_sed=sanitized_inputs["cloud_f_sed"],
            eddy_diffusion_coefficients=sanitized_inputs["eddy_diffusion_coefficients"],
            haze_factor=haze_factor,
            power_law_opacity_350nm=sanitized_inputs["power_law_opacity_350nm"],
            power_law_opacity_coefficient=sanitized_inputs[
                "power_law_opacity_coefficient"
            ],
            gray_opacity=sanitized_inputs["gray_opacity"],
            cloud_photosphere_median_optical_depth=sanitized_inputs[
                "cloud_photosphere_median_optical_depth"
            ],
            patchy_clouds=sanitized_inputs["patchy_clouds"],
            return_cloud_contribution=False,
            additional_absorption_opacities_function=sanitized_inputs[
                "additional_absorption_opacities_function"
            ],
            additional_scattering_opacities_function=sanitized_inputs[
                "additional_scattering_opacities_function"
            ],
            compute_clear_opacities=False,
            return_opacities=False,
            target_particle_radii=sanitized_inputs["target_particle_radii"],
            target_pressure=sanitized_inputs["target_pressure"],
            additional_opacity_functions_kwargs=additional_opacity_functions_kwargs,
        )

        if auto_anisotropic_cloud_scattering:
            self._anisotropic_cloud_scattering = "auto"

        opacities_rosseland = rosseland.compute_rosseland_opacities(
            frequency_bins_edges=self._frequency_bins_edges,
            temperatures=temperatures,
            weights_gauss=self._lines_loaded_opacities["weights_gauss"],
            opacities=opacities[:, :, :1, :],
            continuum_opacities_scattering=continuum_opacities_scattering,
            scattering_in_emission=self._scattering_in_emission,
        )

        opacities_planck = radiative_transfer_core.compute_planck_opacities(
            opacities[:, :, :1, :],
            temperatures,
            self._lines_loaded_opacities["weights_gauss"],
            self._frequency_bins_edges,
            self._scattering_in_emission,
            continuum_opacities_scattering,
        )

        return opacities_rosseland, opacities_planck
        # TODO: confirm that pRT3 and pRT4 agree!

    @_compile_spectral_function(_TRANSIT_RADII_STATIC_ARGNAMES)
    def calculate_transit_radii(
        self,
        temperatures: ArrayLike,
        mass_fractions: dict[str, ArrayLike],
        mean_molar_masses: ArrayLike,
        reference_gravity: float,
        reference_pressure: float,
        planet_radius: float,
        variable_gravity: bool = True,
        opaque_cloud_top_pressure: float = None,
        cloud_particle_mean_radii: dict[str, ArrayLike] = None,
        cloud_particle_radius_distribution_std: float | dict[str, ArrayLike] = None,
        cloud_particle_radius_distribution: str = "lognormal",
        cloud_hansen_a: dict[str, ArrayLike] = None,
        cloud_hansen_b: dict[str, ArrayLike] = None,
        cloud_particle_number_density_grid: dict[str, ArrayLike] = None,
        clouds_particle_porosity_factor: dict[str, float] = None,
        cloud_f_sed: float = None,
        eddy_diffusion_coefficients: float = None,
        haze_factor: float = 1.0,
        power_law_opacity_350nm: float = None,
        power_law_opacity_coefficient: float = None,
        gray_opacity: float = None,
        cloud_fraction: float = 1.0,
        patchy_clouds: list[str] | tuple[str] = None,
        additional_absorption_opacities_function: Callable = None,
        additional_scattering_opacities_function: Callable = None,
        frequencies_to_wavelengths: bool = True,
        return_contribution: bool = False,
        return_clear_spectrum: bool = False,
        return_cloud_contribution: bool = False,
        return_radius_hydrostatic_equilibrium: bool = False,
        return_opacities: bool = False,
        return_abundances: bool = False,
        return_particle_radii_bins: bool = False,
        return_particle_radii: bool = False,
        return_particle_densities: bool = False,
        target_particle_radii: ArrayLike = None,
        target_pressure: ArrayLike = None,
        pressures: ArrayLike = None,
        validate_inputs: bool = True, # Validate it used in the _compile_spectral_function
                                      # decorator, through a wrapper, so don't remove it here,
                                      # even if it is grayed out and appears to not be used.
        make_analytical_clouds_patchy=False,
        additional_opacity_functions_kwargs: dict = None,
    ) -> tuple[ArrayLike, ArrayLike, dict[str, Any]]:
        """Method to calculate the atmosphere's transmission radius
        (for the transmission spectrum).

            Args:
                temperatures:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                mass_fractions:
                    dictionary of mass fractions for all atmospheric absorbers.
                    Dictionary keys are the species names.
                    Every mass fraction array
                    has same length as pressure array.
                mean_molar_masses:
                    the atmospheric mean molecular weight in amu,
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                reference_gravity (float):
                    Surface gravity in cgs at reference radius and pressure.
                reference_pressure (float):
                    Reference pressure P0 in bar where R(P=P0) = R_pl,
                    where R_pl is the reference radius (parameter of this
                    method), and g(P=P0) = gravity, where gravity is the
                    reference gravity (parameter of this method)
                planet_radius (float):
                    Reference radius R_pl, in cm.
                variable_gravity (Optional[bool]):
                    Standard is ``True``. If ``False`` the gravity will be
                    constant as a function of pressure, during the transmission
                    radius calculation.
                opaque_cloud_top_pressure (Optional[float]):
                    Pressure, in bar, where opaque cloud deck is added to the
                    absorption opacity.
                cloud_particle_mean_radii (Optional):
                    dictionary of mean particle radii for all cloud species.
                    Dictionary keys are the cloud species names.
                    Every radius array has same length as pressure array.
                cloud_particle_radius_distribution_std:
                    Width of the log-normal cloud particle size distribution.
                    May be a shared scalar or a per-species dictionary whose
                    values are scalars or layer profiles.
                cloud_particle_radius_distribution (Optional[string]):
                    The cloud particle size distribution to use.
                    Can be either 'lognormal' (default) or 'hansen'.
                    If hansen, the cloud_hansen_b parameters must be used.
                cloud_hansen_a (Optional[dict]):
                    A dictionary of the 'a' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. Equivalent to radius arg.
                    If cloud_hansen_a is not included and dist is "hansen", then it will
                    be computed using Kzz and fsed (recommended).
                    See Nasedkin et al. (2024) for an explanation of how this is done
                    (A&A 687, A298 (2024); doi:10.1051/0004-6361/202449328, Appendix).
                cloud_hansen_b (Optional[dict]):
                    A dictionary of the 'b' parameter values for each
                    included cloud species and for each atmospheric layer,
                    formatted as the kzz argument. This is the width of the hansen
                    distribution normalized by the particle area (1/cloud_hansen_a^2)
                cloud_particle_number_density_grid (Optional[dict]):
                    EXPERIMENTAL # TODO Annie: add description
                clouds_particle_porosity_factor (Optional[dict]):
                    A dictionary of porosity factors depending on the cloud species. This can be useful when opacities
                    are calculated using the Distribution of Hollow Spheres (DHS) method.
                cloud_f_sed (Optional[float]):
                    cloud settling parameter
                eddy_diffusion_coefficients (Optional):
                    the atmospheric eddy diffusion coefficient in cgs
                    (i.e. :math:`\\rm cm^2/s`),
                    at each atmospheric layer
                    (1-d numpy array, same length as pressure array).
                haze_factor (Optional[float]):
                    Scalar factor, increasing the gas Rayleigh scattering
                    cross-section.
                power_law_opacity_350nm (Optional[float]):
                    Scattering opacity at 0.35 micron, in cgs units (cm^2/g).
                power_law_opacity_coefficient (Optional[float]):
                    Has to be given if kappa_zero is defined, this is the
                    wavelength powerlaw index of the parametrized scattering
                    opacity.
                gray_opacity (Optional[float]):
                    Gray opacity value, to be added to the opacity at all
                    pressures and wavelengths (units :math:`\\rm cm^2/g`)
                cloud_fraction:
                    Mix a column without cloud with a column with cloud to the requested fraction
                    (0 = clear, 1 = full cloud cover). Runtime-only cloud sources such as
                    power-law clouds and opaque cloud decks follow the same mixing.
                    By default, assume a full cloud cover.
                patchy_clouds:
                    List of cloud species affected by ``cloud_fraction``.
                additional_absorption_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an absorption opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    This must not be used to add atomic / molecular line opacities in low-resolution mode (c-k),
                    because line opacities require a proper correlated-k treatment.
                    It may be used to add simple cloud absorption laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                    This is a static (compile-time) argument: pass a stable function reference.
                    Passing a freshly created lambda or closure (e.g., the product of a function factory)
                    on each call forces a full recompilation of the spectral kernel.
                    Relatedly, make sure that your function is jax.jit compilable.
                additional_scattering_opacities_function (Optional[function]):
                    A python function that takes wavelength arrays in microns and pressure arrays in bars
                    as input, and returns an isotropic scattering opacity matrix in units of cm^2/g, in the shape of
                    number of wavelength points x number of pressure points.
                    This opacity will then be added to the atmospheric absorption opacity.
                    It may be used to add simple cloud scattering laws, for example, which
                    have opacities that vary only slowly with wavelength, such that the current
                    model resolution is sufficient to resolve any variations.
                    This is a static (compile-time) argument: pass a stable function reference.
                    Passing a freshly created lambda or closure (e.g., the product of a function factory)
                    on each call forces a full recompilation of the spectral kernel.
                    Relatedly, make sure that your function is jax.jit compilable.
                additional_opacity_functions_kwargs (Optional[dict]):
                    Dictionary of extra keyword arguments that is unpacked into both
                    additional_absorption_opacities_function and additional_scattering_opacities_function,
                    on top of the wavelengths and pressures those always receive. If you supply both
                    functions, each of them must be able to accept every key in this dictionary.
                    In contrast to the functions themselves, the values in this dictionary are traced:
                    use it to pass the parameters of your opacity model, so that changing them neither
                    triggers a recompilation nor breaks differentiability.
                frequencies_to_wavelengths (Optional[bool]):
                    If True, convert the frequencies (Hz) output to wavelengths (cm).
                return_contribution (Optional[bool]):
                    If True the transmission and emission contribution function is calculated.
                return_clear_spectrum (Optional[bool]):
                    If True, return the clear spectrum in addition to a cloudy spectrum.
                return_cloud_contribution (Optional[bool]):
                    If True, calculate and return the cloud contribution.
                return_radius_hydrostatic_equilibrium (Optional[bool]):
                    If True, return the radius at hydrostatic equilibrium of the planet.
                return_opacities (Optional[bool]):
                    If True, return the absorption opacities and scattering opacities.
                return_abundances (Optional[bool]):
                    If True, return the mass fractions.
                return_particle_radii_bins (Optional[bool]):
                    If True, return the particles radii bins.
                return_particle_radii (Optional[bool]):
                    If True, return the particles radii.
                return_particle_densities (Optional[bool]):
                    If True, return the particles densities.
                target_particle_radii (Optional[ArrayLike]:
                    Particle radius to target for selective cloud bin disabling. Radius bin closest
                    the input value will be disabled.
                target_pressure (Optional[ArrayLike]:
                    Pressure to target for selective cloud bin disabling. Pressure bin closest to
                    the input value will be disabled.
                validate_inputs (Optional[bool]):
                    If True (default), check the reference gravity is positive.
                    Under ``jax.jit`` this check needs a device-to-host callback
                    on every evaluation, which is costly on GPU). Set to False in
                    retrievals or if you've already checked the inputs.
                pressures (Optional[ArrayLike]):
                    If not None, these pressures will be used as the vertical structure
                    describing the atmosphere. Pressures must be in units of bar and
                    must be sorted in increasing order. They will be converted to cgs
                    units inside pRT.
                make_analytical_clouds_patchy (Optional[bool]):
                    False by default. If True, analytical clouds (opaque cloud top pressure,
                    power law opacity) will be treated as patchy clouds.
            """

        # Check whether user-specified pressure array is tp be used
        solver_pressures = self._resolve_solver_pressures(pressures)

        (
            anisotropic_cloud_scattering,
            patchy_clouds,
            sum_opacities,
        ) = self._init_spectral_function(
            is_emission=False,
            patchy_clouds=patchy_clouds,
        )

        return self._calculate_transit_radii_full_kernel(
            pressures=solver_pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=_convert_scalar_to_strong_type(reference_gravity),
            reference_pressure=_convert_scalar_to_strong_type(reference_pressure),
            planet_radius=_convert_scalar_to_strong_type(planet_radius),
            variable_gravity=variable_gravity,
            opaque_cloud_top_pressure=_convert_scalar_to_strong_type(opaque_cloud_top_pressure),
            cloud_particle_mean_radii=cloud_particle_mean_radii,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            haze_factor=_convert_scalar_to_strong_type(haze_factor),
            power_law_opacity_350nm=_convert_scalar_to_strong_type(power_law_opacity_350nm),
            power_law_opacity_coefficient=_convert_scalar_to_strong_type(power_law_opacity_coefficient),
            gray_opacity=_convert_scalar_to_strong_type(gray_opacity),
            cloud_fraction=_convert_scalar_to_strong_type(cloud_fraction),
            patchy_clouds=patchy_clouds,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            frequencies_to_wavelengths=frequencies_to_wavelengths,
            return_contribution=return_contribution,
            return_clear_spectrum=return_clear_spectrum,
            return_radius_hydrostatic_equilibrium=return_radius_hydrostatic_equilibrium,
            return_cloud_contribution=return_cloud_contribution,
            return_opacities=return_opacities,
            return_abundances=return_abundances,
            return_particle_radii_bins=return_particle_radii_bins,
            return_particle_radii=return_particle_radii,
            return_particle_densities=return_particle_densities,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
            sum_opacities=sum_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            make_analytical_clouds_patchy=make_analytical_clouds_patchy,
            additional_opacity_functions_kwargs=additional_opacity_functions_kwargs,
        )

    @jit(
        static_argnames=[
            "cloud_particle_radius_distribution",
            "variable_gravity",
            "frequencies_to_wavelengths",
            "patchy_clouds",
            "sum_opacities",
            "anisotropic_cloud_scattering",
            "additional_absorption_opacities_function",
            "additional_scattering_opacities_function",
            "return_contribution",
            "return_clear_spectrum",
            "return_radius_hydrostatic_equilibrium",
            "return_cloud_contribution",
            "return_opacities",
            "return_abundances",
            "return_particle_radii_bins",
            "return_particle_radii",
            "return_particle_densities",
            "make_analytical_clouds_patchy",
        ]
    )
    def _calculate_transit_radii_full_kernel(
        self,
        temperatures,
        mass_fractions,
        mean_molar_masses,
        reference_gravity,
        reference_pressure,
        planet_radius,
        pressures=None,
        variable_gravity=True,
        opaque_cloud_top_pressure=None,
        cloud_particle_mean_radii=None,
        cloud_particle_radius_distribution_std=None,
        cloud_particle_radius_distribution="lognormal",
        cloud_hansen_a=None,
        cloud_hansen_b=None,
        cloud_particle_number_density_grid=None,
        clouds_particle_porosity_factor=None,
        cloud_f_sed=None,
        eddy_diffusion_coefficients=None,
        haze_factor=1.0,
        power_law_opacity_350nm=None,
        power_law_opacity_coefficient=None,
        gray_opacity=None,
        cloud_photosphere_median_optical_depth=None,
        cloud_fraction=1.0,
        patchy_clouds=None,
        additional_absorption_opacities_function=None,
        additional_scattering_opacities_function=None,
        frequencies_to_wavelengths=True,
        return_contribution=False,
        return_clear_spectrum=False,
        return_radius_hydrostatic_equilibrium=False,
        return_cloud_contribution=False,
        return_opacities=False,
        return_abundances=False,
        return_particle_radii_bins=False,
        return_particle_radii=False,
        return_particle_densities=False,
        target_particle_radii=None,
        target_pressure=None,
        sum_opacities=False,
        anisotropic_cloud_scattering=None,
        make_analytical_clouds_patchy=False,
        additional_opacity_functions_kwargs=None,
    ):
        """JIT-compiled kernel for calculate_transit_radii. See calculate_transit_radii for parameter docs."""

        sanitized_inputs = _sanitize_flux_opacity_inputs(
            patchy_clouds=patchy_clouds,
            cloud_particle_mean_radii=cloud_particle_mean_radii,
            cloud_hansen_a=cloud_hansen_a,
            cloud_hansen_b=cloud_hansen_b,
            clouds_particle_porosity_factor=clouds_particle_porosity_factor,
            cloud_f_sed=cloud_f_sed,
            eddy_diffusion_coefficients=eddy_diffusion_coefficients,
            opaque_cloud_top_pressure=opaque_cloud_top_pressure,
            cloud_particle_radius_distribution_std=cloud_particle_radius_distribution_std,
            power_law_opacity_350nm=power_law_opacity_350nm,
            power_law_opacity_coefficient=power_law_opacity_coefficient,
            gray_opacity=gray_opacity,
            cloud_photosphere_median_optical_depth=None,
            cloud_particle_number_density_grid=cloud_particle_number_density_grid,
            additional_absorption_opacities_function=additional_absorption_opacities_function,
            additional_scattering_opacities_function=additional_scattering_opacities_function,
            target_particle_radii=target_particle_radii,
            target_pressure=target_pressure,
            cloud_species=self._cloud_species,
            missing_float=_MISSING_FLOAT,
            missing_neg_inf=_MISSING_NEG_INF,
            additional_opacities_placeholder_fn=_additional_opacities_placeholder_fn,
        )

        partial_coverage_cloud_species = self._get_partial_coverage_cloud_species(
            sanitized_inputs["patchy_clouds"]
        )

        has_partial_coverage_analytical_cloud = (
            self._has_partial_coverage_analytical_cloud(
                opaque_cloud_top_pressure=opaque_cloud_top_pressure,
                power_law_opacity_350nm=power_law_opacity_350nm,
                power_law_opacity_coefficient=power_law_opacity_coefficient,
                make_analytical_clouds_patchy=make_analytical_clouds_patchy,
            )
        )

        # Determine whether clear opacities are needed.
        if (
            len(partial_coverage_cloud_species) > 0
            or has_partial_coverage_analytical_cloud
        ):
            should_compute_opacities_without_patchy_clouds = True
        else:
            should_compute_opacities_without_patchy_clouds = False

        (
            opacities,
            continuum_opacities_scattering,
            cloud_anisotropic_scattering_opacities,
            cloud_absorption_opacities,
            cloud_opacities,
            cloud_particle_mean_radii,
            cloud_scattering_in_transmission,
            clear_opacities,
            clear_continuum_opacities_scattering,
            clear_cloud_anisotropic_scattering_opacities,
            clear_cloud_absorption_opacities,
        ) = self._calculate_opacities(
            pressures=pressures,
            temperatures=temperatures,
            mass_fractions=mass_fractions,
            mean_molar_masses=mean_molar_masses,
            reference_gravity=reference_gravity,
            opaque_cloud_top_pressure=sanitized_inputs["opaque_cloud_top_pressure"],
            cloud_particle_mean_radii=sanitized_inputs["cloud_particle_mean_radii"],
            cloud_particle_radius_distribution_std=sanitized_inputs[
                "cloud_particle_radius_distribution_std"
            ],
            cloud_particle_radius_distribution=cloud_particle_radius_distribution,
            cloud_hansen_a=sanitized_inputs["cloud_hansen_a"],
            cloud_hansen_b=sanitized_inputs["cloud_hansen_b"],
            cloud_particle_number_density_grid=sanitized_inputs[
                "cloud_particle_number_density_grid"
            ],
            clouds_particle_porosity_factor=sanitized_inputs[
                "clouds_particle_porosity_factor"
            ],
            cloud_f_sed=sanitized_inputs["cloud_f_sed"],
            eddy_diffusion_coefficients=sanitized_inputs["eddy_diffusion_coefficients"],
            haze_factor=haze_factor,
            power_law_opacity_350nm=sanitized_inputs["power_law_opacity_350nm"],
            power_law_opacity_coefficient=sanitized_inputs[
                "power_law_opacity_coefficient"
            ],
            gray_opacity=sanitized_inputs["gray_opacity"],
            cloud_photosphere_median_optical_depth=sanitized_inputs[
                "cloud_photosphere_median_optical_depth"
            ],
            patchy_clouds=sanitized_inputs["patchy_clouds"],
            return_cloud_contribution=return_cloud_contribution,
            additional_absorption_opacities_function=sanitized_inputs[
                "additional_absorption_opacities_function"
            ],
            additional_scattering_opacities_function=sanitized_inputs[
                "additional_scattering_opacities_function"
            ],
            compute_clear_opacities=should_compute_opacities_without_patchy_clouds,
            return_opacities=return_opacities,
            sum_opacities=sum_opacities,
            anisotropic_cloud_scattering=anisotropic_cloud_scattering,
            target_particle_radii=sanitized_inputs["target_particle_radii"],
            target_pressure=sanitized_inputs["target_pressure"],
            make_analytical_clouds_patchy=make_analytical_clouds_patchy,
            additional_opacity_functions_kwargs=additional_opacity_functions_kwargs,
        )

        # Calculate the transit radii
        transit_radii, radius_hydrostatic_equilibrium, transmission_contribution = (
            self._calculate_transit_radii(
                pressures=pressures,
                temperatures=temperatures,
                mean_molar_masses=mean_molar_masses,
                reference_gravity=reference_gravity,
                reference_pressure=reference_pressure,
                planet_radius=planet_radius,
                variable_gravity=variable_gravity,
                opacities=opacities,
                continuum_opacities_scattering=continuum_opacities_scattering,
                sum_opacities=sum_opacities,
                return_contributions=return_contribution,
            )
        )

        # --- Cloud blending for partial coverage ---
        transit_radii_clear = transit_radii # Will be filled with actually clear(er) spectrum below
                                            #   if needed.
        if should_compute_opacities_without_patchy_clouds:
            if return_clear_spectrum:
                # Always compute clear(er) radii when the caller requested it.
                transit_radii_clear, _, _ = self._calculate_transit_radii(
                    pressures=pressures,
                    temperatures=temperatures,
                    mean_molar_masses=mean_molar_masses,
                    reference_gravity=reference_gravity,
                    reference_pressure=reference_pressure,
                    planet_radius=planet_radius,
                    variable_gravity=variable_gravity,
                    opacities=clear_opacities,
                    continuum_opacities_scattering=clear_continuum_opacities_scattering,
                    sum_opacities=sum_opacities,
                    return_contributions=False,
                )

                def _blend_partial_cloud_transit(args):
                    return args[2] * args[0] + (1 - args[2]) * args[1]

                def _pass_through_radii(args):
                    return args[0]

                transit_radii = lax.cond(
                    jnp.less(cloud_fraction, 1),
                    _blend_partial_cloud_transit,
                    _pass_through_radii,
                    (transit_radii, transit_radii_clear, cloud_fraction),
                )
            else:
                # Only compute clear radii and blend when cloud_fraction < 1.
                def _blend_partial_cloud_transit(cloudy_radii):
                    clear_radii, _, _ = self._calculate_transit_radii(
                        pressures=pressures,
                        temperatures=temperatures,
                        mean_molar_masses=mean_molar_masses,
                        reference_gravity=reference_gravity,
                        reference_pressure=reference_pressure,
                        planet_radius=planet_radius,
                        variable_gravity=variable_gravity,
                        opacities=clear_opacities,
                        continuum_opacities_scattering=clear_continuum_opacities_scattering,
                        sum_opacities=sum_opacities,
                        return_contributions=False,
                    )
                    return (
                        cloud_fraction * cloudy_radii
                        + (1 - cloud_fraction) * clear_radii
                    )

                def _pass_through_cloud_radii(radii):
                    return radii

                transit_radii = lax.cond(
                    jnp.less(cloud_fraction, 1),
                    _blend_partial_cloud_transit,
                    _pass_through_cloud_radii,
                    transit_radii,
                )

        # --- Build additional outputs ---
        additional_outputs = {}

        if return_particle_radii_bins:
            additional_outputs["particle_radii_bins"] = self._clouds_loaded_opacities[
                "particle_radii_bins"
            ]

        if return_particle_radii:
            additional_outputs["particle_radii"] = self._clouds_loaded_opacities[
                "particle_radii"
            ]

        if (
            sanitized_inputs["cloud_particle_mean_radii"] is not None
            or len(sanitized_inputs["cloud_f_sed"]) > 0
        ):
            additional_outputs["cloud_particle_mean_radii"] = cloud_particle_mean_radii

        if return_particle_densities:
            additional_outputs["particle_densities"] = self._clouds_loaded_opacities[
                "particle_densities"
            ]

        if return_contribution:
            additional_outputs["transmission_contribution"] = transmission_contribution

        if return_clear_spectrum:
            additional_outputs["clear_spectrum"] = transit_radii_clear

        if return_radius_hydrostatic_equilibrium:
            additional_outputs["radius_hydrostatic_equilibrium"] = (
                radius_hydrostatic_equilibrium
            )

        if return_cloud_contribution and len(self._cloud_species) > 0:
            additional_outputs["cloud_contribution"] = (
                clouds.compute_species_optical_depths(
                    reference_gravity=reference_gravity,
                    pressures=pressures,
                    cloud_opacities=cloud_opacities,
                )
            )

        if return_opacities:
            additional_outputs["opacities"] = opacities
            additional_outputs["continuum_opacities_scattering"] = (
                continuum_opacities_scattering
            )
            additional_outputs["cloud_absorption_opacities"] = (
                cloud_absorption_opacities
            )
            additional_outputs["cloud_anisotropic_scattering_opacities"] = (
                cloud_anisotropic_scattering_opacities # TODO Paul make sure the right opacities are
                                                       #    returned here when transmission spectra are
                                                       #    calculated (namely, (1-g) should not be applied).
            )

        if return_abundances:
            additional_outputs["abundances"] = mass_fractions

        if frequencies_to_wavelengths:
            return (
                frequency2wavelength(self._frequencies),
                transit_radii,
                additional_outputs,
            )
        else:
            return self._frequencies, transit_radii, additional_outputs

    def get_wavelengths(self):
        """Convert the frequency grid into wavelengths in centimeter."""
        # Not a property because the child SpectralModel already have a 'wavelengths' attribute that behaves differently
        return frequency2wavelength(self._frequencies)

    def load_all_opacities(self):
        print("Loading Radtrans opacities...")

        # Load line opacities
        self.load_line_opacities(self._path_input_data)

        # Read continuum opacities
        # CIA
        if len(self._gas_continuum_contributors) > 0:
            self.load_cia_opacities(self._path_input_data)

        # Clouds
        if len(self._cloud_species) > 0:
            # Inherited from ReadOpacities in _read_opacities.py
            self.load_cloud_opacities(self._path_input_data)

        print("Successfully loaded all opacities")

    def load_cia_opacities(self, path_input_data):
        for collision in self._cias_loaded_opacities:
            # Just checks if collision == 'H-' here.
            # A bit overkill to handle it like this, since the
            # hminus subpackage should only ever concern H-.
            if collision in hminus.get_non_cia_gas_continuum_contributions():
                continue

            # Checks for inversions (e.g. He--H2 instead of H2--He, both of
            # which should be valid).
            hdf5_file = cia.get_cia_aliases(collision)
            hdf5_file = CIAOpacity.find(
                path_input_data=path_input_data, species=hdf5_file
            )

            print(
                f" Loading CIA opacities for {collision} from file '{hdf5_file}'...",
                end="",
            )

            (
                self._cias_loaded_opacities[collision]["molecules"],
                self._cias_loaded_opacities[collision]["weight"],
                self._cias_loaded_opacities[collision]["lambda"],
                self._cias_loaded_opacities[collision]["temperature"],
                self._cias_loaded_opacities[collision]["alpha"],
            ) = load.load_cia_opacity(hdf5_file)

            print(" Done.")

        print(" Successfully loaded all CIA opacities")

    def load_cloud_opacities(self, path_input_data):
        """Load the cloud opacities, given all species listed in
        self._cloud_species.

        Args:
            path_input_data: path to petitRADTRANS input_data directory

        Returns:
            Nothing, but fills Radtrans's _clouds_loaded_opacities dictionary.
        """

        # Will contain paths to input files for all cloud species
        hdf5_files = []
        # For every species, this would reduce
        # path/Al2-O3-NatAbund(s)_crystalline_000__DHS.R39_0.1-250mu.cotable.petitRADTRANS.h5
        # to crystalline_000, for example.
        internal_structures = []
        # For every species, this will contain
        # DHS or Mie
        scattering_methods = []

        for i in range(len(self._cloud_species)):
            # Get the cloud species full name
            hdf5_file = CloudOpacity.get_aliases(self._cloud_species[i])

            # Return the absolute path to the cloud species input data.
            hdf5_file = CloudOpacity.find(
                path_input_data=path_input_data, species=hdf5_file
            )

            internal_structures.append(hdf5_file.rsplit(")_", 1)[1].split("__", 1)[0])
            scattering_methods.append(hdf5_file.rsplit("__", 1)[1].split(".", 1)[0])

            hdf5_files.append(hdf5_file)

        clouds_particle_densities = jnp.zeros(
            len(hdf5_files), dtype=FEAUTRIER_JAX_DTYPE
        )
        clouds_absorption_opacities = None
        clouds_scattering_opacities = None
        clouds_asymmetry_parameters = None
        cloud_wavelengths = None
        clouds_particle_radii_bins = None
        clouds_particle_radii = None

        for i, hdf5_file in enumerate(hdf5_files):
            print(
                f" Loading opacities of cloud species '{self._cloud_species[i]}' from file '{hdf5_file}' "
                f"({internal_structures[i]}, using {scattering_methods[i]} scattering)...",
                end="",
            )

            (
                _cloud_wavelengths,
                _cloud_particle_radii_bins,
                _cloud_particle_radii,
                _cloud_particle_densities,
                _cloud_absorption_opacities,
                _cloud_scattering_opacities,
                _cloud_asymmetry_parameters,
            ) = load.load_cloud_opacity(hdf5_file)

            if i == 0:
                # Initialize cloud arrays
                # This means that all cloud species must share the same wavelength,
                # and cloud particle radii!
                cloud_wavelengths = _cloud_wavelengths
                clouds_particle_radii_bins = _cloud_particle_radii_bins
                clouds_particle_radii = _cloud_particle_radii

                clouds_absorption_opacities = jnp.zeros(
                    (
                        clouds_particle_radii.size,
                        cloud_wavelengths.size,
                        len(hdf5_files),
                    ),
                    dtype=FEAUTRIER_JAX_DTYPE,
                )
                clouds_scattering_opacities = jnp.zeros(
                    (
                        clouds_particle_radii.size,
                        cloud_wavelengths.size,
                        len(hdf5_files),
                    ),
                    dtype=FEAUTRIER_JAX_DTYPE,
                )
                clouds_asymmetry_parameters = jnp.zeros(
                    (
                        clouds_particle_radii.size,
                        cloud_wavelengths.size,
                        len(hdf5_files),
                    ),
                    dtype=FEAUTRIER_JAX_DTYPE,
                )

            clouds_particle_densities = clouds_particle_densities.at[i].set(
                _cloud_particle_densities
            )
            clouds_absorption_opacities = clouds_absorption_opacities.at[:, :, i].set(
                _cloud_absorption_opacities
            )
            clouds_scattering_opacities = clouds_scattering_opacities.at[:, :, i].set(
                _cloud_scattering_opacities
            )
            clouds_asymmetry_parameters = clouds_asymmetry_parameters.at[:, :, i].set(
                _cloud_asymmetry_parameters
            )

            print(" Done.")

        # Flip wavelengths/wavenumbers axis to match wavelengths ordering
        clouds_absorption_opacities = clouds_absorption_opacities[:, ::-1, :]
        clouds_scattering_opacities = clouds_scattering_opacities[:, ::-1, :]
        clouds_asymmetry_parameters = clouds_asymmetry_parameters[:, ::-1, :]

        clouds_absorption_opacities = clouds_absorption_opacities.at[
            clouds_absorption_opacities < 0.0
        ].set(0.0)
        clouds_scattering_opacities = clouds_scattering_opacities.at[
            clouds_scattering_opacities < 0.0
        ].set(0.0)

        self._clouds_loaded_opacities["particle_densities"] = jnp.array(
            clouds_particle_densities, dtype=FEAUTRIER_JAX_DTYPE
        )
        self._clouds_loaded_opacities["absorption_opacities"] = jnp.array(
            clouds_absorption_opacities, dtype=FEAUTRIER_JAX_DTYPE
        )
        self._clouds_loaded_opacities["scattering_opacities"] = jnp.array(
            clouds_scattering_opacities, dtype=FEAUTRIER_JAX_DTYPE
        )
        self._clouds_loaded_opacities["particle_asymmetry_parameters"] = jnp.array(
            clouds_asymmetry_parameters, dtype=FEAUTRIER_JAX_DTYPE
        )
        self._clouds_loaded_opacities["wavelengths"] = jnp.array(
            cloud_wavelengths, dtype=FEAUTRIER_JAX_DTYPE
        )
        self._clouds_loaded_opacities["particle_radii_bins"] = jnp.array(
            clouds_particle_radii_bins, dtype=FEAUTRIER_JAX_DTYPE
        )
        self._clouds_loaded_opacities["particle_radii"] = jnp.array(
            clouds_particle_radii, dtype=FEAUTRIER_JAX_DTYPE
        )

        print(" Successfully loaded all clouds opacities")

    def load_line_opacities(self, path_input_data):
        """Load the line opacities for spectral calculation.
        The default pressure-temperature grid is a log-uniform (10, 13) grid.

        Args:
            path_input_data: path to petitRADTRANS input_data directory
        """
        import h5py

        if self._line_opacity_mode == "c-k":
            category = "correlated_k_opacities"
        elif self._line_opacity_mode == "lbl":
            category = "line_by_line_opacities"
        else:
            raise ValueError(
                f"invalid line opacity mode: '{self._line_opacity_mode}' (must be 'c-k'|'lbl')"
            )

        host_dtype = np.dtype(WORKING_JAX_DTYPE)
        temperature_pressure_grids = {}
        temperature_grid_sizes = {}
        pressure_grid_sizes = {}
        opacity_grids = {}
        g_gauss = None
        weights_gauss = None

        # Read opacities grid
        if len(self._line_species) > 0:
            for i, species in enumerate(self._line_species):
                hdf5_file = load.get_line_opacity_file(
                    path_input_data=path_input_data, species=species, category=category
                )

                # Load g grid for correlated-k
                if self._line_opacity_mode == "c-k" and i == 0:
                    with h5py.File(hdf5_file, "r") as f:
                        g_gauss = f["samples"][:]
                        weights_gauss = f["weights"][:]

                # Load temperature-pressure grid
                (
                    temperature_pressure_grids[species],
                    temperature_grid_sizes[species],
                    pressure_grid_sizes[species],
                ) = load.load_line_opacities_pressure_temperature_grid(
                    hdf5_file=hdf5_file
                )
                # Load the opacities
                print(
                    f" Loading line opacities of species '{species}' from file '{hdf5_file}'...",
                    end="",
                )

                if self._line_opacity_mode == "c-k":
                    opacity_grids[species] = np.asarray(
                        load.load_hdf5_ktables(
                            file_path_hdf5=hdf5_file,
                            frequencies=self._frequencies,
                            g_size=g_gauss.size,
                            temperature_pressure_grid_size=temperature_pressure_grids[
                                species
                            ].shape[0],
                            frequency_grid_misalignment_tolerance=self._frequency_grid_misalignment_tolerance,
                            return_host_array=True
                        ),
                        dtype=host_dtype,
                    )
                elif self._line_opacity_mode == "lbl":
                    opacity_grids[species] = load.load_hdf5_line_opacity_table(
                        file_path_hdf5=hdf5_file,
                        frequencies=self._frequencies,
                        line_by_line_opacity_sampling=self._line_by_line_opacity_sampling,
                    )

                print(" Done.")

            if g_gauss is not None:
                self._lines_loaded_opacities["g_gauss"] = jnp.array(
                    g_gauss,
                    dtype=WORKING_JAX_DTYPE,
                )
                self._lines_loaded_opacities["weights_gauss"] = jnp.array(
                    weights_gauss,
                    dtype=WORKING_JAX_DTYPE,
                )

            self._stack_loaded_line_opacities(
                temperature_pressure_grids=temperature_pressure_grids,
                temperature_grid_sizes_map=temperature_grid_sizes,
                pressure_grid_sizes_map=pressure_grid_sizes,
                opacity_grids=opacity_grids,
            )
            self._set_unstacked_line_opacity_debug_data(
                temperature_pressure_grids=temperature_pressure_grids,
                temperature_grid_sizes=temperature_grid_sizes,
                pressure_grid_sizes=pressure_grid_sizes,
                opacity_grids=opacity_grids,
            )

            print(" Successfully loaded all line opacities")


tree_util.register_pytree_node(
    Radtrans, Radtrans._tree_flatten, Radtrans._tree_unflatten
)
