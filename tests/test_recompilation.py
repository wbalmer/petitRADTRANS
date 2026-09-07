"""Ensure repeated spectral calculations do not re-trace or recompile.

The public calculate_flux / calculate_transit_radii methods delegate to jitted
kernels; once compiled for a given configuration, repeated calls with the same
shapes -- including calls that vary only the Python representation of scalar
inputs (float vs numpy vs jnp scalars) -- must hit the jit cache.
"""

import contextlib
import logging

import jax
import jax.numpy as jnp
import numpy as np

from .context import petitRADTRANS
from .utils import (
    test_parameters,
    temperature_guillot_2010,
    temperature_isothermal,
)


def _init_radtrans_correlated_k():
    return petitRADTRANS.radtrans.Radtrans(
        pressures=test_parameters["pressures"],
        line_species=test_parameters["spectrum_parameters"][
            "line_species_correlated_k"
        ],
        rayleigh_species=test_parameters["spectrum_parameters"]["rayleigh_species"],
        gas_continuum_contributors=test_parameters["spectrum_parameters"][
            "continuum_opacities"
        ],
        cloud_species=list(test_parameters["cloud_parameters"]["cloud_species"].keys()),
        wavelength_boundaries=test_parameters["spectrum_parameters"][
            "wavelength_range_correlated_k"
        ],
        line_opacity_mode="c-k",
        scattering_in_emission=False,
    )


atmosphere_ck = _init_radtrans_correlated_k()


class _CompileLogCounter(logging.Handler):
    """Collects XLA compilation log records emitted under jax_log_compiles."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        message = record.getMessage()

        if message.startswith("Compiling jit("):
            self.messages.append(message)


@contextlib.contextmanager
def _count_compiles():
    counter = _CompileLogCounter()
    logger = logging.getLogger("jax")
    previous_level = logger.level
    previous_log_compiles = jax.config.jax_log_compiles
    jax.config.update("jax_log_compiles", True)
    logger.addHandler(counter)
    logger.setLevel(logging.DEBUG)

    try:
        yield counter
    finally:
        logger.removeHandler(counter)
        logger.setLevel(previous_level)
        jax.config.update("jax_log_compiles", previous_log_compiles)


def _flux_arguments(**overrides):
    arguments = dict(
        temperatures=temperature_guillot_2010,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        reference_gravity=test_parameters["planetary_parameters"][
            "reference_gravity"
        ],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        frequencies_to_wavelengths=False,
    )
    arguments.update(overrides)

    return arguments


def _transit_arguments(**overrides):
    arguments = dict(
        temperatures=temperature_isothermal,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        reference_gravity=test_parameters["planetary_parameters"][
            "reference_gravity"
        ],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        planet_radius=test_parameters["planetary_parameters"]["radius"]
        * petitRADTRANS.physical_constants.r_jup_mean,
        reference_pressure=test_parameters["planetary_parameters"][
            "reference_pressure"
        ],
        frequencies_to_wavelengths=False,
    )
    arguments.update(overrides)

    return arguments


def test_calculate_flux_does_not_recompile_on_repeat_calls():
    atmosphere_ck.calculate_flux(**_flux_arguments())  # warm-up compile

    with _count_compiles() as counter:
        atmosphere_ck.calculate_flux(**_flux_arguments())

    assert counter.messages == []


def test_calculate_flux_scalar_representations_share_compilation():
    reference_gravity = test_parameters["planetary_parameters"]["reference_gravity"]
    atmosphere_ck.calculate_flux(**_flux_arguments())  # warm-up compile

    with _count_compiles() as counter:
        for gravity, haze_factor, cloud_fraction in (
            (reference_gravity, 1.0, 1.0),
            (np.float64(reference_gravity), np.float64(1.0), 1),
            (jnp.asarray(reference_gravity), jnp.float32(1.0), np.int64(1)),
        ):
            atmosphere_ck.calculate_flux(
                **_flux_arguments(
                    reference_gravity=gravity,
                    haze_factor=haze_factor,
                    cloud_fraction=cloud_fraction,
                )
            )

    # Converting a not-yet-seen scalar dtype may compile a trivial eager
    # primitive (convert_element_type, ...) once; the spectral kernels must
    # never recompile.
    kernel_compiles = [
        message for message in counter.messages if "calculate" in message
    ]

    assert kernel_compiles == []


def test_calculate_transit_radii_does_not_recompile_on_repeat_calls():
    atmosphere_ck.calculate_transit_radii(**_transit_arguments())  # warm-up compile

    with _count_compiles() as counter:
        atmosphere_ck.calculate_transit_radii(**_transit_arguments())
        atmosphere_ck.calculate_transit_radii(
            **_transit_arguments(haze_factor=np.float64(1.0), cloud_fraction=1)
        )

    kernel_compiles = [
        message for message in counter.messages if "calculate" in message
    ]

    assert kernel_compiles == []


def test_radtrans_treedef_is_hashable():
    treedef = jax.tree_util.tree_structure(atmosphere_ck)

    assert isinstance(hash(treedef), int)

    leaves, treedef_roundtrip = jax.tree_util.tree_flatten(atmosphere_ck)
    rebuilt = jax.tree_util.tree_unflatten(treedef_roundtrip, leaves)

    assert rebuilt._cloud_species == tuple(atmosphere_ck._cloud_species)
    assert rebuilt._line_species == tuple(atmosphere_ck._line_species)


def test_additional_opacity_function_is_accepted_and_cached():
    atmosphere_ck.calculate_flux(
        **_flux_arguments(
            additional_absorption_opacities_function=(
                petitRADTRANS.radtrans._additional_opacities_placeholder_fn
            )
        )
    )  # warm-up compile with the callback as a static argument

    with _count_compiles() as counter:
        atmosphere_ck.calculate_flux(
            **_flux_arguments(
                additional_absorption_opacities_function=(
                    petitRADTRANS.radtrans._additional_opacities_placeholder_fn
                )
            )
        )

    assert counter.messages == []


def test_invalid_reference_gravity_raises():
    for calculate, arguments in (
        (atmosphere_ck.calculate_flux, _flux_arguments),
        (atmosphere_ck.calculate_transit_radii, _transit_arguments),
    ):
        try:
            calculate(**arguments(reference_gravity=-1.0))
        except ValueError as error:
            assert "reference gravity must be > 0" in str(error)
        else:
            raise AssertionError(
                "expected ValueError for negative reference gravity"
            )


def test_unsorted_call_time_pressures_raise():
    unsorted_pressures = test_parameters["pressures"][::-1]

    for calculate, arguments in (
        (atmosphere_ck.calculate_flux, _flux_arguments),
        (atmosphere_ck.calculate_transit_radii, _transit_arguments),
    ):
        try:
            calculate(**arguments(pressures=unsorted_pressures))
        except ValueError as error:
            assert "increasing order" in str(error)
        else:
            raise AssertionError(
                "expected ValueError for pressures not in increasing order"
            )
