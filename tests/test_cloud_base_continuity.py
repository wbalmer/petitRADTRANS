"""Regression tests for the continuity of the cloud parameterisation.

The cloud profile used to be truncated at its base with ``jnp.where(pressures <
cloud_base, ...)``. Sampled on a fixed pressure grid that makes the profile a
*step* function of the cloud-base pressure: as the base crosses a grid point,
that layer flips between zero and the full base abundance. Gradient-based
optimisers and samplers then see an infinite derivative at every layer boundary,
and settle exactly on one. These tests pin the smooth replacement.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from petitRADTRANS.chemistry import clouds as fc  # noqa: E402
from petitRADTRANS.chemistry.core import calculate_cloud_mass_fractions  # noqa: E402
from petitRADTRANS.jax_dtype import FEAUTRIER_JAX_DTYPE  # noqa: E402


# petitRADTRANS picks its working dtype at import time, so a suite that imported
# it before this module gets single precision. Scale the probes accordingly
# rather than depending on import order.
_DOUBLE_PRECISION = FEAUTRIER_JAX_DTYPE == jnp.float64
# How far either side of a grid boundary the profile is probed for a step.
STEP_PROBE = 1e-7 if _DOUBLE_PRECISION else 1e-4
# Central-difference step and tolerance for the gradient checks.
GRADIENT_STEP = 1e-6 if _DOUBLE_PRECISION else 1e-3
GRADIENT_RTOL = 1e-4 if _DOUBLE_PRECISION else 2e-2
# What counts as continuous: a step flips a whole layer, so it is never small.
CONTINUITY_RTOL = 1e-6 if _DOUBLE_PRECISION else 1e-3

PRESSURES = jnp.logspace(-6, 1, 120)
LOG_PRESSURES = np.log10(np.asarray(PRESSURES))
LAYER_SPACING = float(LOG_PRESSURES[1] - LOG_PRESSURES[0])
# A grid boundary, i.e. the worst case for the old hard cutoff.
BOUNDARY = float(LOG_PRESSURES[45])
CLOUD_ABUNDANCE = 1e-3


def _column(log_cloud_base, f_sed=5.0, smoothing_width=None):
    profile = fc.cloud_vertical_profile(
        PRESSURES,
        cloud_base_pressure=10**log_cloud_base,
        cloud_abundance=CLOUD_ABUNDANCE,
        f_sed=f_sed,
        smoothing_width=smoothing_width,
    )

    return jnp.sum(profile * jnp.diff(PRESSURES, prepend=0.0))


def test_hard_cutoff_still_steps_across_a_grid_boundary():
    """The defect this module fixes, kept explicit so the tests have a control."""
    below = float(_column(BOUNDARY - STEP_PROBE, smoothing_width=0.0))
    above = float(_column(BOUNDARY + STEP_PROBE, smoothing_width=0.0))

    assert abs(above - below) / below > 1.0


def test_smoothed_profile_is_continuous_across_a_grid_boundary():
    below = float(_column(BOUNDARY - STEP_PROBE))
    above = float(_column(BOUNDARY + STEP_PROBE))

    assert abs(above - below) / below < CONTINUITY_RTOL


@pytest.mark.parametrize("offset_in_layers", np.linspace(-2.5, 2.5, 21))
def test_smoothed_profile_gradient_matches_finite_differences(offset_in_layers):
    """Both directions: the original axis scan only probed +delta and missed the cliff."""
    log_cloud_base = BOUNDARY + offset_in_layers * LAYER_SPACING

    autodiff = float(jax.grad(_column)(log_cloud_base))
    finite_difference = float(
        (_column(log_cloud_base + GRADIENT_STEP) - _column(log_cloud_base - GRADIENT_STEP))
        / (2 * GRADIENT_STEP)
    )

    np.testing.assert_allclose(finite_difference, autodiff, rtol=GRADIENT_RTOL)


@pytest.mark.parametrize("log_cloud_base", [BOUNDARY, -1.0, -4.7])
def test_zero_width_reproduces_the_legacy_profile_exactly(log_cloud_base):
    cloud_base = 10**log_cloud_base
    f_sed = 5.0

    legacy = jnp.where(
        PRESSURES < cloud_base,
        CLOUD_ABUNDANCE * (PRESSURES / cloud_base) ** f_sed,
        0.0,
    )
    smoothed = fc.cloud_vertical_profile(
        PRESSURES,
        cloud_base_pressure=cloud_base,
        cloud_abundance=CLOUD_ABUNDANCE,
        f_sed=f_sed,
        smoothing_width=0.0,
    )

    np.testing.assert_array_equal(np.asarray(smoothed), np.asarray(legacy))


def test_smoothing_leaves_the_profile_above_the_base_untouched():
    """The taper is local: only the layers within a few widths of the base move."""
    cloud_base = 10**BOUNDARY
    kwargs = dict(
        cloud_base_pressure=cloud_base, cloud_abundance=CLOUD_ABUNDANCE, f_sed=5.0
    )

    smoothed = np.asarray(fc.cloud_vertical_profile(PRESSURES, **kwargs))
    legacy = np.asarray(
        fc.cloud_vertical_profile(PRESSURES, smoothing_width=0.0, **kwargs)
    )

    # Four layers above the base is roughly eight smoothing widths.
    well_above_base = LOG_PRESSURES < BOUNDARY - 4 * LAYER_SPACING
    np.testing.assert_allclose(
        smoothed[well_above_base], legacy[well_above_base], rtol=1e-4
    )


@pytest.mark.parametrize("f_sed", [0.5, 2.0, 10.0, 20.0])
def test_profile_and_gradient_stay_finite_for_every_f_sed(f_sed):
    """The taper is steepened by f_sed, so the sub-base tail always decays.

    The legacy form evaluates ``(P / P_base) ** f_sed`` on every layer before
    discarding the deep ones, which reaches 1e87 at ``f_sed = 20`` and overflows
    single precision to a NaN gradient.
    """
    profile = fc.cloud_vertical_profile(
        PRESSURES,
        cloud_base_pressure=10**BOUNDARY,
        cloud_abundance=CLOUD_ABUNDANCE,
        f_sed=f_sed,
    )
    gradient = jax.grad(_column)(BOUNDARY, f_sed)

    assert bool(jnp.all(jnp.isfinite(profile)))
    assert bool(jnp.isfinite(gradient))
    assert float(jnp.max(profile)) <= CLOUD_ABUNDANCE


def test_depletion_factor_is_smooth_and_complements_the_taper():
    factor = lambda log_base: jnp.sum(  # noqa: E731
        fc.cloud_base_depletion_factor(PRESSURES, 10**log_base)
    )

    autodiff = float(jax.grad(factor)(BOUNDARY))
    finite_difference = float(
        (factor(BOUNDARY + GRADIENT_STEP) - factor(BOUNDARY - GRADIENT_STEP))
        / (2 * GRADIENT_STEP)
    )

    np.testing.assert_allclose(finite_difference, autodiff, rtol=GRADIENT_RTOL)

    at_base = fc.cloud_base_depletion_factor(
        jnp.array([10**BOUNDARY]), 10**BOUNDARY
    )
    np.testing.assert_allclose(np.asarray(at_base), 0.5, rtol=1e-12)


def test_equilibrium_cloud_base_is_continuous_in_temperature():
    """The base used to be snapped to a layer midpoint, i.e. a staircase."""
    from petitRADTRANS.temperature_profiles import (
        guillot_global_temperature_profile,
    )

    def cloud_base(t_equilibrium):
        temperatures = guillot_global_temperature_profile(
            PRESSURES, 1e-2, 1e2, 10**3.5, 400.0, t_equilibrium
        )
        return jnp.log10(
            fc.calculate_cloud_base_pressure_mgsio3(PRESSURES, temperatures, 0.0, 0.55)
        )

    bases = np.array([float(cloud_base(t)) for t in np.linspace(1300.0, 1800.0, 501)])

    # A staircase would take a handful of values separated by a full layer.
    assert len(np.unique(bases)) > 400
    assert np.abs(np.diff(bases)).max() < 0.5 * LAYER_SPACING

    autodiff = float(jax.grad(cloud_base)(1430.0))
    finite_difference = float((cloud_base(1430.01) - cloud_base(1429.99)) / 2e-2)
    np.testing.assert_allclose(finite_difference, autodiff, rtol=GRADIENT_RTOL)


def test_smoothing_width_must_be_static():
    with pytest.raises(TypeError, match="static scalar"):
        jax.jit(
            lambda width: fc.cloud_vertical_profile(
                PRESSURES,
                cloud_base_pressure=1.0,
                cloud_abundance=CLOUD_ABUNDANCE,
                f_sed=2.0,
                smoothing_width=width,
            )
        )(0.03)


@pytest.mark.parametrize(
    "smoothing_parameters, expect_step",
    [
        ({}, False),
        ({"cloud_base_smoothing": 0.0}, True),
        ({"cloud_base_smoothing_StubCloud": 0.0}, True),
        ({"cloud_base_smoothing": 0.0, "cloud_base_smoothing_StubCloud": 0.03}, False),
    ],
)
def test_cloud_base_smoothing_is_resolved_from_parameters(
    monkeypatch, smoothing_parameters, expect_step
):
    """Per-cloud keys override the global one, which overrides the default."""
    import petitRADTRANS.chemistry.core

    class FakeCloudOpacity:
        def __init__(self, *_args, **_kwargs):
            self.species_full_name = "StubCloud_placeholder"

    monkeypatch.setattr(
        petitRADTRANS.chemistry.core, "CloudOpacity", FakeCloudOpacity
    )
    petitRADTRANS.chemistry.core._get_cached_cloud_metadata.cache_clear()

    def column(log_cloud_base):
        parameters = dict(
            smoothing_parameters,
            **{
                "log_Pbase_StubCloud": log_cloud_base,
                "log_X_cb_StubCloud": -3.0,
                "f_sed": 5.0,
            },
        )
        mass_fractions, _, _ = calculate_cloud_mass_fractions(
            pressures=PRESSURES,
            temperatures=jnp.full_like(PRESSURES, 1000.0),
            metallicity=None,
            co_ratio=None,
            cloud_species=("dummy",),
            parameters=parameters,
            mass_fractions={},
            mean_molar_mass=jnp.full_like(PRESSURES, 2.33),
        )
        return float(jnp.sum(mass_fractions["dummy"]))

    below = column(BOUNDARY - STEP_PROBE)
    above = column(BOUNDARY + STEP_PROBE)
    relative_jump = abs(above - below) / below

    petitRADTRANS.chemistry.core._get_cached_cloud_metadata.cache_clear()

    if expect_step:
        assert relative_jump > 1e-2
    else:
        assert relative_jump < CONTINUITY_RTOL


def test_profile_survives_every_jax_transformation():
    """jit, forward and reverse mode, second order, and vmap."""

    def column(log_cloud_base, log_cloud_abundance, f_sed):
        profile = fc.cloud_vertical_profile(
            PRESSURES,
            cloud_base_pressure=10**log_cloud_base,
            cloud_abundance=10**log_cloud_abundance,
            f_sed=f_sed,
        )
        return jnp.sum(profile * jnp.diff(PRESSURES, prepend=0.0))

    arguments = (BOUNDARY, -3.0, 5.0)

    assert bool(jnp.isfinite(jax.jit(column)(*arguments)))
    assert bool(jnp.all(jnp.isfinite(jax.jit(jax.grad(column))(*arguments))))

    forward = jax.jacfwd(column, argnums=(0, 1, 2))(*arguments)
    reverse = jax.jacrev(column, argnums=(0, 1, 2))(*arguments)
    np.testing.assert_allclose(
        [float(value) for value in forward],
        [float(value) for value in reverse],
        rtol=1e-9,
    )

    # Dense-metric HMC and Laplace approximations need the second derivative.
    hessian = jax.jit(jax.hessian(lambda v: column(v[0], v[1], v[2])))(
        jnp.asarray(arguments)
    )
    assert bool(jnp.all(jnp.isfinite(hessian)))
    np.testing.assert_allclose(np.asarray(hessian), np.asarray(hessian).T, rtol=1e-8)

    bases = jnp.linspace(BOUNDARY - 0.2, BOUNDARY + 0.2, 16)
    batched = jax.jit(jax.vmap(jax.grad(column)))(
        bases, jnp.full(16, -3.0), jnp.full(16, 5.0)
    )
    assert bool(jnp.all(jnp.isfinite(batched)))


def test_second_derivative_is_correct_on_a_grid_boundary():
    """The curvature has to survive at exactly the pressures that trapped the MAP."""
    step = 1e-4 if _DOUBLE_PRECISION else 1e-2

    autodiff = float(jax.hessian(_column)(BOUNDARY))
    finite_difference = float(
        (_column(BOUNDARY + step) - 2 * _column(BOUNDARY) + _column(BOUNDARY - step))
        / step**2
    )

    np.testing.assert_allclose(finite_difference, autodiff, rtol=1e-2)


def test_unresolved_f_sed_gives_a_zero_cloud_without_a_nan(monkeypatch):
    """The sentinel is non-finite, so it must not reach the profile arithmetic."""
    import petitRADTRANS.chemistry.core

    class FakeCloudOpacity:
        def __init__(self, *_args, **_kwargs):
            self.species_full_name = "StubCloud_placeholder"

    monkeypatch.setattr(petitRADTRANS.chemistry.core, "CloudOpacity", FakeCloudOpacity)
    petitRADTRANS.chemistry.core._get_cached_cloud_metadata.cache_clear()

    def column(log_cloud_abundance):
        mass_fractions, _, _ = calculate_cloud_mass_fractions(
            pressures=PRESSURES,
            temperatures=jnp.full_like(PRESSURES, 1000.0),
            metallicity=None,
            co_ratio=None,
            cloud_species=("dummy",),
            # No f_sed and no f_sed_<cloud>: the cloud is not configured.
            parameters={
                "log_Pbase_StubCloud": BOUNDARY,
                "log_X_cb_StubCloud": log_cloud_abundance,
            },
            mass_fractions={},
            mean_molar_mass=jnp.full_like(PRESSURES, 2.33),
        )
        return jnp.sum(mass_fractions["dummy"])

    value = column(-3.0)
    gradient = jax.grad(column)(-3.0)

    petitRADTRANS.chemistry.core._get_cached_cloud_metadata.cache_clear()

    assert float(value) == 0.0
    assert bool(jnp.isfinite(gradient))


def test_equilibrium_cloud_base_is_differentiable_in_its_inputs():
    from petitRADTRANS.temperature_profiles import (
        guillot_global_temperature_profile,
    )

    temperatures = guillot_global_temperature_profile(
        PRESSURES, 1e-2, 1e2, 10**3.5, 400.0, 1430.0
    )

    def cloud_base(mean_molar_mass):
        return jnp.log10(
            fc.calculate_cloud_base_pressure_mgsio3(
                PRESSURES, temperatures, 0.0, 0.55, mean_molar_mass
            )
        )

    autodiff = float(jax.jit(jax.grad(cloud_base))(2.33))
    finite_difference = float((cloud_base(2.3301) - cloud_base(2.3299)) / 2e-4)

    np.testing.assert_allclose(finite_difference, autodiff, rtol=1e-3)
    assert bool(jnp.isfinite(jax.hessian(cloud_base)(2.33)))
