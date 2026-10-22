"""Test the additional_absorption/scattering_opacities_function options of Radtrans.

A user function returning a constant opacity kappa must act exactly like the built-in
constant opacities: gray_opacity for absorption, and a power-law cloud with a zero
wavelength exponent for scattering. When made patchy, its spectrum must be the
cloud_fraction-weighted mix of the spectra with and without the function.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from .context import petitRADTRANS
from .utils import test_parameters, temperature_guillot_2010

kappa = 0.05  # cm2/g; large enough to change the spectra noticeably
cloud_fraction = 0.3

atmosphere = petitRADTRANS.radtrans.Radtrans(
    pressures=test_parameters["pressures"],
    line_species=test_parameters["spectrum_parameters"]["line_species_correlated_k"],
    rayleigh_species=test_parameters["spectrum_parameters"]["rayleigh_species"],
    gas_continuum_contributors=test_parameters["spectrum_parameters"]["continuum_opacities"],
    wavelength_boundaries=test_parameters["spectrum_parameters"]["wavelength_range_correlated_k"],
    line_opacity_mode="c-k",
    scattering_in_emission=True,  # otherwise scattering opacities are ignored in emission
)


def constant_opacity(wavelengths, pressures, kappa):
    """Additional opacity function: kappa (passed via additional_opacity_functions_kwargs) everywhere."""
    return jnp.full((wavelengths.size, pressures.size), kappa)


def calculate_spectrum(mode, **kwargs):
    common = dict(
        temperatures=temperature_guillot_2010,
        mass_fractions=test_parameters["mass_fractions_correlated_k"],
        mean_molar_masses=test_parameters["mean_molar_mass"],
        reference_gravity=test_parameters["planetary_parameters"]["reference_gravity"],
    )

    if mode == "emission":
        return np.asarray(atmosphere.calculate_flux(**common, **kwargs)[1]) # [1] returns flux

    return np.asarray(
        atmosphere.calculate_transit_radii(
            **common,
            planet_radius=test_parameters["planetary_parameters"]["radius"]
            * petitRADTRANS.physical_constants.r_jup_mean,
            reference_pressure=test_parameters["planetary_parameters"]["reference_pressure"],
            **kwargs,
        )[1] # [1] returns transit radii
    )


def additional_opacity(kind, **kwargs):
    """Arguments adding constant_opacity as an additional absorption or scattering opacity."""
    return {
        f"additional_{kind}_opacities_function": constant_opacity,
        "additional_opacity_functions_kwargs": {"kappa": kappa},
        **kwargs,
    }


builtin_equivalent = {
    "absorption": dict(gray_opacity=kappa),
    "scattering": dict(power_law_opacity_350nm=kappa, power_law_opacity_coefficient=0.0),
}

# These decorators will run the test functions below looping through mode values = emission, then transmission.
# The second decorator will do the same for the kind values. It is like nested loops and 4 tests will be run.
@pytest.mark.parametrize("mode", ["emission", "transmission"])
@pytest.mark.parametrize("kind", ["absorption", "scattering"])
def test_additional_opacity_matches_builtin_equivalent(mode, kind):
    spectrum = calculate_spectrum(mode, **additional_opacity(kind))

    assert not np.allclose(spectrum, calculate_spectrum(mode))  # the opacity has an effect...
                                                                # since calculate_spectrum(mode) will
                                                                # be cloud-free
    np.testing.assert_allclose(  # ...which is the one of the matching built-in opacity
        spectrum, calculate_spectrum(mode, **builtin_equivalent[kind]), rtol=1e-6
    )


def test_additional_scattering_differs_from_absorption_in_emission_only():
    for mode, same in (("emission", False), ("transmission", True)):
        absorption = calculate_spectrum(mode, **additional_opacity("absorption"))
        scattering = calculate_spectrum(mode, **additional_opacity("scattering"))

        assert np.allclose(absorption, scattering, rtol=1e-6) == same


@pytest.mark.parametrize("mode", ["emission", "transmission"])
@pytest.mark.parametrize("kind", ["absorption", "scattering"])
@pytest.mark.parametrize("patchy", [True, False])
def test_additional_opacity_patchiness(mode, kind, patchy):
    patchy_kwargs = additional_opacity(
        kind, cloud_fraction=cloud_fraction, **{f"make_additional_{kind}_opacities_patchy": patchy}
    )

    if patchy:
        spectrum = calculate_spectrum(mode, **patchy_kwargs)
    else:  # nothing is patchy, so the user is warned that cloud_fraction has no effect
        with pytest.warns(UserWarning, match="cloud_fraction < 1"):
            spectrum = calculate_spectrum(mode, **patchy_kwargs)

    with_opacity = calculate_spectrum(mode, **additional_opacity(kind))

    if patchy:
        expected = cloud_fraction * with_opacity + (1 - cloud_fraction) * calculate_spectrum(mode)
    else:  # a non-patchy opacity covers the whole planet, whatever the cloud_fraction
        expected = with_opacity

    np.testing.assert_allclose(spectrum, expected, rtol=1e-6)
