"""Generate the static transmission explorer data used in the getting started docs."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

cst = import_module("petitRADTRANS.physical_constants")
Radtrans = import_module("petitRADTRANS.radtrans").Radtrans


OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "_static"
    / "getting_started_transmission_explorer_data.json"
)

LINE_SPECIES_RES = ("H2O__POKAZATEL.R400", "CO-NatAbund.R400", "CH4__MM.R400", "CO2.R400", "Na__NewAllard.R400", "K__Allard.R400")
LINE_SPECIES = ("H2O", "CO-NatAbund", "CH4", "CO2", "Na", "K")

RAYLEIGH_SPECIES = ("H2", "He")
GAS_CONTINUUM_CONTRIBUTORS = ("H2--H2", "H2--He")
WAVELENGTH_BOUNDARIES = (0.5, 12.0)

BACKGROUND_H2_FRACTION = 0.74 / (0.74 + 0.24)
BACKGROUND_HE_FRACTION = 0.24 / (0.74 + 0.24)

SPECIES_ORDER = ("H2O", "CO-NatAbund", "CO2", "CH4", "Na", "K")
SPECIES_LABELS = {
    "H2O": "H2O",
    "CO-NatAbund": "CO",
    "CO2": "CO2",
    "CH4": "CH4",
    "Na": "Na",
    "K": "K",
    "H2": "H2",
    "He": "He",
}
SPECIES_LABELS_OPAS = {
    "H2O__POKAZATEL": "H2O",
    "CO-NatAbund": "CO-NatAbund",
    "CO2": "CO2",
    "CH4__MM": "CH4",
    "Na__NewAllard": "Na",
    "K__Allard": "K",
    "H2": "H2",
    "He": "He",
}

SPECIES_COLORS = {
    "H2O": "#2f7ed8",
    "CO-NatAbund": "#f28f43",
    "CO2": "#2c9c74",
    "CH4": "#d94f45",
    "Na": "#7b62c0",
    "K": "#9d6b53",
    "H2": "#5f6b7a",
    "He": "#a5adb8",
}
MOLAR_MASSES = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O": 18.01528,
    "CO-NatAbund": 28.0101,
    "CO2": 44.0095,
    "CH4": 16.0425,
    "Na": 22.989769,
    "K": 39.0983,
}

CONTROL_SPECS = {
    "temperature_top": {
        "label": "T top [K]",
        "group": "temperature",
        "min": 400.0,
        "max": 2500.0,
        "step": 50.0,
        "default": 1200.0,
        "decimals": 0,
        "grid": [400.0, 800.0, 1200.0, 1800.0, 2500.0],
    },
    "temperature_middle": {
        "label": "T middle [K]",
        "group": "temperature",
        "min": 400.0,
        "max": 2500.0,
        "step": 50.0,
        "default": 1200.0,
        "decimals": 0,
        "grid": [400.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0],
    },
    "temperature_bottom": {
        "label": "T deep [K]",
        "group": "temperature",
        "min": 400.0,
        "max": 3500.0,
        "step": 50.0,
        "default": 1200.0,
        "decimals": 0,
        "grid": [400.0, 1200.0, 2000.0, 2750.0, 3500.0],
    },
    "log_h2o": {
        "label": "log10 X(H2O)",
        "group": "abundance",
        "min": -10.0,
        "max": -1.0,
        "step": 0.25,
        "default": -3.0,
        "decimals": 2,
        "grid": [-10.0, -6.0, -3.0, -2.0, -1.5, -1.0],
    },
    "log_co": {
        "label": "log10 X(CO)",
        "group": "abundance",
        "min": -10.0,
        "max": -1.0,
        "step": 0.25,
        "default": -2.0,
        "decimals": 2,
        "grid": [-10.0, -6.0, -2.0, -1.75, -1.5, -1.25, -1.0],
    },
    "log_co2": {
        "label": "log10 X(CO2)",
        "group": "abundance",
        "min": -12.0,
        "max": -1.0,
        "step": 0.25,
        "default": -4.0,
        "decimals": 2,
        "grid": [-12.0, -8.0, -4.0, -2.0, -1.0],
    },
    "log_ch4": {
        "label": "log10 X(CH4)",
        "group": "abundance",
        "min": -12.0,
        "max": -1.0,
        "step": 0.25,
        "default": -5.0,
        "decimals": 2,
        "grid": [-12.0, -9.0, -5.0, -3.0, -1.0],
    },
    "log_na": {
        "label": "log10 X(Na)",
        "group": "abundance",
        "min": -12.0,
        "max": -1.0,
        "step": 0.25,
        "default": -4.0,
        "decimals": 2,
        "grid": [-12.0, -8.0, -4.0, -2.0, -1.0],
    },
    "log_k": {
        "label": "log10 X(K)",
        "group": "abundance",
        "min": -12.0,
        "max": -1.0,
        "step": 0.25,
        "default": -6.0,
        "decimals": 2,
        "grid": [-12.0, -9.0, -6.0, -3.0, -1.0],
    },
    "log10_surface_gravity": {
        "label": "log10 g [cgs]",
        "group": "planet",
        "min": 2.0,
        "max": 4.5,
        "step": 0.05,
        "default": 3.5,
        "decimals": 2,
        "grid": [2.0, 2.75, 3.5, 4.0, 4.5],
    },
    "planet_radius_r_jup": {
        "label": "Planet radius [R_Jup]",
        "group": "planet",
        "min": 0.7,
        "max": 2.0,
        "step": 0.02,
        "default": 1.0,
        "decimals": 2,
        "grid": [0.7, 0.85, 1.0, 1.2, 1.4, 2.0],
    },
}

DEFAULT_PARAMETERS = {
    name: spec["default"]
    for name, spec in CONTROL_SPECS.items()
}
DEFAULT_PARAMETERS["reference_pressure"] = 0.01


def build_radtrans() -> Radtrans:
    return Radtrans(
        pressures=jnp.logspace(-6, 2, 800),
        line_species=LINE_SPECIES_RES,
        rayleigh_species=RAYLEIGH_SPECIES,
        gas_continuum_contributors=GAS_CONTINUUM_CONTRIBUTORS,
        wavelength_boundaries=WAVELENGTH_BOUNDARIES,
    )


def build_temperature_profile(
    pressure_bars: np.ndarray,
    temperature_top: float,
    temperature_middle: float,
    temperature_bottom: float,
) -> jnp.ndarray:
    temperature_nodes = np.array(
        [temperature_top, temperature_middle, temperature_bottom],
        dtype=float,
    )
    node_pressures_bar = np.array(
        [pressure_bars.min(), 1e-2, pressure_bars.max()],
        dtype=float,
    )
    temperatures = np.interp(
        np.log10(pressure_bars),
        np.log10(node_pressures_bar),
        temperature_nodes,
    )
    return jnp.asarray(temperatures)


def build_mass_fraction_profiles(
    pressure_shape: tuple[int, ...],
    log_h2o: float,
    log_co: float,
    log_co2: float,
    log_ch4: float,
    log_na: float,
    log_k: float,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    trace_mass_fractions = {
        "H2O__POKAZATEL": 10.0 ** log_h2o,
        "CO-NatAbund": 10.0 ** log_co,
        "CO2": 10.0 ** log_co2,
        "CH4__MM": 10.0 ** log_ch4,
        "Na__NewAllard": 10.0 ** log_na,
        "K__Allard": 10.0 ** log_k,
    }

    remaining_background = 1.0 - sum(trace_mass_fractions.values())
    if remaining_background <= 0.0:
        raise ValueError("trace-gas mass fractions must sum to less than one")

    mass_fraction_profiles = {
        species: jnp.full(pressure_shape, value)
        for species, value in trace_mass_fractions.items()
    }
    mass_fraction_profiles["H2"] = jnp.full(
        pressure_shape,
        remaining_background * BACKGROUND_H2_FRACTION,
    )
    mass_fraction_profiles["He"] = jnp.full(
        pressure_shape,
        remaining_background * BACKGROUND_HE_FRACTION,
    )

    combined_mass_fractions = {
        **trace_mass_fractions,
        "H2": remaining_background * BACKGROUND_H2_FRACTION,
        "He": remaining_background * BACKGROUND_HE_FRACTION,
    }
    inverse_mean_molar_mass = sum(
        value / MOLAR_MASSES[SPECIES_LABELS_OPAS[species]]
        for species, value in combined_mass_fractions.items()
    )
    mean_molar_mass = 1.0 / inverse_mean_molar_mass

    return mass_fraction_profiles, jnp.full(pressure_shape, mean_molar_mass)


def calculate_transmission_state(
    radtrans: Radtrans,
    pressure_bars: np.ndarray,
    parameters: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    temperatures = build_temperature_profile(
        pressure_bars=pressure_bars,
        temperature_top=parameters["temperature_top"],
        temperature_middle=parameters["temperature_middle"],
        temperature_bottom=parameters["temperature_bottom"],
    )
    mass_fractions, mean_molar_masses = build_mass_fraction_profiles(
        pressure_shape=radtrans.pressures.shape,
        log_h2o=parameters["log_h2o"],
        log_co=parameters["log_co"],
        log_co2=parameters["log_co2"],
        log_ch4=parameters["log_ch4"],
        log_na=parameters["log_na"],
        log_k=parameters["log_k"],
    )
    wavelengths, transit_radii, _ = radtrans.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=10.0 ** parameters["log10_surface_gravity"],
        planet_radius=parameters["planet_radius_r_jup"] * cst.r_jup_mean,
        reference_pressure=parameters["reference_pressure"],
    )

    return (
        np.asarray(wavelengths, dtype=float) * 1e4,
        np.asarray(transit_radii, dtype=float) / cst.r_jup_mean,
    )


def compute_dataset() -> dict[str, object]:
    radtrans = build_radtrans()
    pressure_bars = np.asarray(radtrans.pressures, dtype=float) * 1e-6
    log_pressure_bars = np.log10(pressure_bars)
    temperature_node_pressures_bar = np.array(
        [pressure_bars.min(), 1e-2, pressure_bars.max()],
        dtype=float,
    )

    baseline_wavelengths, baseline_spectrum = calculate_transmission_state(
        radtrans=radtrans,
        pressure_bars=pressure_bars,
        parameters=DEFAULT_PARAMETERS,
    )

    responses: dict[str, dict[str, object]] = {}
    for parameter_name, spec in CONTROL_SPECS.items():
        spectra_r_jup = []
        for value in spec["grid"]:
            parameter_values = dict(DEFAULT_PARAMETERS)
            parameter_values[parameter_name] = float(value)
            wavelengths, spectrum = calculate_transmission_state(
                radtrans=radtrans,
                pressure_bars=pressure_bars,
                parameters=parameter_values,
            )
            if not np.allclose(wavelengths, baseline_wavelengths):
                raise RuntimeError(
                    f"wavelength grid changed while varying {parameter_name}"
                )
            spectra_r_jup.append(np.round(spectrum, 10).tolist())

        responses[parameter_name] = {
            "grid": [float(grid_value) for grid_value in spec["grid"]],
            "spectra_r_jup": spectra_r_jup,
        }

    return {
        "pressure_bars": np.round(pressure_bars, 12).tolist(),
        "log_pressure_bars": np.round(log_pressure_bars, 12).tolist(),
        "temperature_node_pressures_bar": np.round(
            temperature_node_pressures_bar,
            12,
        ).tolist(),
        "temperature_node_log_pressures": np.round(
            np.log10(temperature_node_pressures_bar),
            12,
        ).tolist(),
        "wavelengths_microns": np.round(baseline_wavelengths, 10).tolist(),
        "baseline_spectrum_r_jup": np.round(baseline_spectrum, 10).tolist(),
        "control_specs": CONTROL_SPECS,
        "species_order": list(SPECIES_ORDER),
        "species_labels": SPECIES_LABELS,
        "species_colors": SPECIES_COLORS,
        "molar_masses": MOLAR_MASSES,
        "background_fractions": {
            "H2": BACKGROUND_H2_FRACTION,
            "He": BACKGROUND_HE_FRACTION,
        },
        "default_parameters": DEFAULT_PARAMETERS,
        "responses": responses,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = compute_dataset()
    OUTPUT_PATH.write_text(
        json.dumps(dataset, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()