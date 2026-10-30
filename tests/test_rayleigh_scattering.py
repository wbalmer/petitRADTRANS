"""Numerical and literature-anchor checks; no external opacity data required."""

import hashlib
import os
import shutil

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from petitRADTRANS import _input_data
from petitRADTRANS import physical_constants as cst
from petitRADTRANS.config.configuration import get_input_data_subpaths, petitradtrans_config_parser
from petitRADTRANS.opacities.opacities import Opacity
from petitRADTRANS.opacities.rayleigh import compute_rayleigh_scattering_opacities
from petitRADTRANS.opacities.write import write_h2_rayleigh_opacities
from petitRADTRANS.radtrans_core import rayleigh_scattering as rayleigh


MASSES = dict(H2=2, He=4, H2O=18, CO2=44, O2=32, N2=28, CO=28, CH4=16)
# Sneep & Ubachs (2005) Table 2 measurement wavenumber, and reference density.
SNEEP_UBACHS_WAVELENGTH = 1e8 / 18788.4
H2_TABLE_SHA256 = "b6828930ced3b4befe0911074a83d448a3a685ff594fada733b469bbd42708eb"


def inputs(species, dtype):
    def a(values):
        return jnp.asarray(values, dtype=dtype)
    return (a([0., 1e-12, 1e-4, 1., 1.]),
            a([1500., 2500., 3500., 5000., 6500., 10000., 15000., 30000., 41354., 100000.]),
            a([MASSES[species]] * 5), a([300., 150., 1000., 2500., 300.]), a([1e6, 1e-4, 1e3, 1e6, 0.]))


def kappa(species, wavelengths_um, temperature=300.):
    """Opacity per gram of species, as a NumPy array over wavelength."""
    wavelengths = jnp.asarray(np.atleast_1d(wavelengths_um) * 1e4)
    return np.asarray(rayleigh.compute_rayleigh_scattering(
        species, jnp.ones(1), wavelengths, jnp.array([2.33]), jnp.array([temperature]), jnp.array([1e3])
    ))[:, 0]


def peck_huang_sigma(wavelength_um):
    """Cross section implied by Peck & Huang (1977) at 273.15 K and 1 atm, without a King factor."""
    nm1 = 0.0148956 / (180.7 - wavelength_um**-2) + 0.0049037 / (92. - wavelength_um**-2)
    n2 = (1. + nm1)**2
    return 24 * np.pi**3 * (1e4 / wavelength_um)**4 / 2.6867811e19**2 * ((n2 - 1) / (n2 + 2))**2


@pytest.mark.parametrize("species", MASSES)
def test_float32_matches_float64_and_zero_abundance_is_differentiable(species):
    with jax.enable_x64(True):
        reference = np.asarray(rayleigh.compute_rayleigh_scattering(species, *inputs(species, jnp.float64)))
    # Test with x64 genuinely disabled, including compile-time weak constants.
    with jax.enable_x64(False):
        args = inputs(species, jnp.float32)

        def compute(x):
            return rayleigh.compute_rayleigh_scattering(species, x, *args[1:])
        result = compute(args[0])
        assert result.dtype == jnp.float32
        assert np.all(np.isfinite(result))
        np.testing.assert_array_equal(result[:, 0], 0.)
        assert np.all(np.asarray(result[:, 1:]) > 0.)
        np.testing.assert_allclose(result, reference, rtol=4e-6, atol=0.)
        grad = jax.grad(lambda x: compute(x).sum())(args[0])
        assert np.all(np.isfinite(grad))
        assert np.all(np.asarray(grad) > 0.)
        # The right derivative at X=0 must approach kappa(epsilon)/epsilon.
        epsilon = 1e-4
        near_zero = compute(args[0].at[0].set(epsilon))[:, 0].sum() / epsilon
        np.testing.assert_allclose(grad[0], near_zero, rtol=2e-5)


@pytest.mark.parametrize("species", ["H2", "H2O"])
def test_temperature_and_abundance_gradients_match_finite_differences(species):
    with jax.enable_x64(True):
        def evaluate(parameters):
            x, t = parameters
            return rayleigh.compute_rayleigh_scattering(
                species, jnp.array([x]), jnp.array([5000.]), jnp.array([2.33]),
                jnp.array([t]), jnp.array([2e6]),
            )[0, 0]
        for temperature in (500., 2000.):
            point = np.array([0.03, temperature])
            gradient = np.asarray(jax.grad(evaluate)(jnp.array(point)))
            steps = np.array([1e-5, 0.1])
            for i, step in enumerate(steps):
                delta = np.zeros(2)
                delta[i] = step
                difference = float(evaluate(point + delta) - evaluate(point - delta)) / (2 * step)
                np.testing.assert_allclose(gradient[i], difference, rtol=2e-5, atol=1e-20)
        assert np.all(np.isfinite(jax.hessian(evaluate)(jnp.array([0., 500.]))))


def test_no_density_dependence():
    # Sneep & Ubachs (2005) Eqs. 2--4: the cross section is independent of density.
    with jax.enable_x64(True):
        for species in MASSES:
            def at_pressure(pressure, mass_fraction):
                return rayleigh.compute_rayleigh_scattering(
                    species, jnp.array([mass_fraction]), jnp.array([5000.]), jnp.array([2.33]),
                    jnp.array([300.]), jnp.array([pressure]),
                )
            np.testing.assert_allclose(at_pressure(1., 1.), at_pressure(1e8, 1.), rtol=1e-14)


@pytest.mark.parametrize("species,measured,n_based", [
    ("N2", 5.10, 5.30), ("CO", 6.19, 6.82), ("CO2", 12.4, 13.29),
])
def test_sneep_ubachs_table_2_n_based_cross_sections(species, measured, n_based):
    with jax.enable_x64(True):
        sigma = kappa(species, SNEEP_UBACHS_WAVELENGTH * 1e-4)[0] * MASSES[species] * cst.amu * 1e27
    # The paper derives CO from its rounded n-1, hence the wider tolerance.
    np.testing.assert_allclose(sigma, n_based, rtol=5e-3 if species == "CO" else 1e-3)
    assert sigma > measured


def test_sneep_ubachs_table_2_refractivities_expose_the_printed_equation_errors():
    with jax.enable_x64(True):
        lamb_inv = jnp.array(18788.4)
        # Table 2 lists these to three figures. Eqs. 13 and 17 as printed give 428 and 229.5.
        assert round(float(rayleigh._co2_refractivity(lamb_inv)) * 1e6) == 427
        assert round(float(rayleigh._co_refractivity(lamb_inv)) * 1e6) == 325


def test_carbon_monoxide_matches_smith_refractivity():
    # Smith et al. (1976) at 273.15 K, as tabulated at refractiveindex.info. Their shortest
    # wavelength, 167.2 nm, is just outside the range in which the refractivity is evaluated.
    with jax.enable_x64(True):
        nm1 = float(rayleigh._co_refractivity(jnp.array(1e4 / 0.2894)))
        np.testing.assert_allclose(nm1 * 2.6867811e19 / cst.standard_number_density_for_refractive_index,
                                   3.632e-4, rtol=5e-3)


def test_nitrogen_mass_and_infrared_tail():
    with jax.enable_x64(True):
        # Sneep & Ubachs (2005), Table 2: n-based sigma=5.30e-27 cm^2 at 532.2 nm.
        result = rayleigh.n2_rayleigh(jnp.array([5322., 50000., 100000.]), jnp.ones(1))[:, 0]
        np.testing.assert_allclose(result[0] * (28 * cst.amu), 5.30e-27, rtol=1e-3)
        assert result[1] > result[2] > 0
        np.testing.assert_allclose(result[1] / result[2], 16., rtol=1e-3)
        boundary = 1e8 / 4860.
        sides = rayleigh.n2_rayleigh(jnp.array([boundary - 0.01, boundary + 0.01]), jnp.ones(1))
        np.testing.assert_allclose(sides[0], sides[1], rtol=5e-6)


def test_helium_published_polarizability():
    with jax.enable_x64(True):
        # Puchalski et al. (2016), Table VII.
        np.testing.assert_allclose(rayleigh.helium_polarizability(jnp.array([6329.908])),
                                   1.39181197, rtol=1e-8)
        values = rayleigh.helium_polarizability(jnp.array([3000., 5000., 10000., 1e8]))
        assert np.all(np.diff(values) < 0.)
        np.testing.assert_allclose(values[-1], 1.38376077, rtol=1e-8)
        # Mansfield & Peck (1969) at 273.15 K, where the old polynomial was 36% low at 0.6 um.
        for wavelength_um in (0.3, 0.5, 0.6, 0.9, 1.5):
            nm1 = 0.01470091 / (423.98 - wavelength_um**-2)
            n2 = (1. + nm1)**2
            reference = (24 * np.pi**3 * (1e4 / wavelength_um)**4 / 2.6867811e19**2
                         * ((n2 - 1) / (n2 + 2))**2)
            np.testing.assert_allclose(kappa("He", wavelength_um)[0] * 4 * cst.amu, reference, rtol=5e-3)


def test_methane_is_below_measured_extinction():
    with jax.enable_x64(True):
        # He et al. (2021) Eq. 10 at 532.2 nm, and the Sneep & Ubachs (2005) measured extinction.
        sigma = kappa("CH4", SNEEP_UBACHS_WAVELENGTH * 1e-4)[0] * 16 * cst.amu * 1e27
        np.testing.assert_allclose(sigma, 11.352, rtol=1e-3)
        assert sigma < 12.47


def test_egan_water_refractivity_anchors():
    with jax.enable_x64(True):
        water = rayleigh.water_molar_refractivity(jnp.array([6329.908, 15423.83]), jnp.array([303.]))[:, 0]
        # Egan & Yang (2024), and Egan (2026), section 7.3.
        np.testing.assert_allclose(water[0], 3.7466, rtol=5e-4)
        np.testing.assert_allclose(water[1] / water[0], 0.97569, rtol=3e-4)
        # Egan & Yang (2024) slope, of the opposite sign to the IAPWS fit.
        hot, cold = (rayleigh.water_molar_refractivity(jnp.array([6329.908]), jnp.array([t]))[0, 0]
                     for t in (433., 293.))
        np.testing.assert_allclose((hot / cold - 1.) / 140., 1.5e-6, rtol=2e-2)


def test_no_jumps_or_poles_in_wavelength_or_temperature():
    with jax.enable_x64(True):
        # Includes the former window edges, and the CO2 and O2 poles.
        wavelengths = np.geomspace(0.12, 30., 20000)
        temperatures = jnp.asarray(np.linspace(100., 5000., 4901))
        for species in MASSES:
            trend = kappa(species, wavelengths) * wavelengths**4
            assert np.all(np.isfinite(trend))
            assert np.abs(np.diff(np.log(trend))).max() < 3e-3, species
            thermal = np.asarray(rayleigh.compute_rayleigh_scattering(
                species, jnp.ones_like(temperatures), jnp.array([5000.]), jnp.ones_like(temperatures),
                temperatures, jnp.ones_like(temperatures),
            ))[0]
            assert np.abs(np.diff(np.log(thermal))).max() < 1e-3, species


@pytest.mark.parametrize("dtype,x64", [(jnp.float32, False), (jnp.float64, True)])
def test_carbon_dioxide_pole_is_not_evaluated(dtype, x64):
    with jax.enable_x64(x64):
        pole = 1e4 / 2418.136
        wavelengths = jnp.asarray(np.geomspace(pole - 2e-5, pole + 2e-5, 41) * 1e4, dtype=dtype)

        def total(temperature_unused, x):
            return rayleigh.co2_rayleigh(wavelengths, x).sum()
        result = rayleigh.co2_rayleigh(wavelengths, jnp.ones(1, dtype=dtype))
        assert np.all(np.isfinite(result))
        assert float(result.max() / result.min()) < 1.001
        assert np.isfinite(jax.grad(total, argnums=1)(0., jnp.ones(1, dtype=dtype))).all()


def _h2_table_file():
    return rayleigh._get_h2_table_file(petitradtrans_config_parser.get_input_data_path())


def _full_h2_table():
    with h5py.File(_h2_table_file(), "r") as table:
        return dict(
            photon_energy_ev=table["wavenumbers"][:] * rayleigh.HC_EV_UM * 1e-4,
            rotational_n=table["rotational_quantum_numbers"][:],
            vibrational_v=table["vibrational_quantum_numbers"][:],
            excitation_energy_ev=table["excitation_energies"][:],
            sigma_over_energy4_mb=table["elastic_reduced_cross_sections"][:],
            raman_over_energy4_mb=table["raman_reduced_cross_sections"][:],
        )


def test_h2_table_is_read_from_the_input_data_directory():
    path_input_data = petitradtrans_config_parser.get_input_data_path()
    assert _h2_table_file() == os.path.join(
        path_input_data, "opacities", "continuum", "rayleigh",
        "1H2__Singor.DeltaWavenumber363_0.207-27.5mu.rayleightable.petitRADTRANS.h5"
    )


def test_h2_table_follows_path_input_data(tmp_path):
    # A table with doubled cross sections shows which directory has actually been read.
    table = _full_h2_table()
    directory = tmp_path / get_input_data_subpaths()["rayleigh_opacities"]
    directory.mkdir(parents=True)
    write_h2_rayleigh_opacities(
        hdf5_file=directory / rayleigh._H2_TABLE_FILE,
        wavenumbers=table["photon_energy_ev"] / (rayleigh.HC_EV_UM * 1e-4),
        elastic_reduced_cross_sections=2. * table["sigma_over_energy4_mb"],
        raman_reduced_cross_sections=2. * table["raman_over_energy4_mb"],
        rotational_quantum_numbers=table["rotational_n"],
        vibrational_quantum_numbers=table["vibrational_v"],
        excitation_energies=table["excitation_energy_ev"],
        anti_stokes_hold_energy=1.,
        channel_count=9582,
        source_archive_md5="",
    )
    with jax.enable_x64(True):
        x, wavelengths, temperatures = jnp.ones(1), jnp.array([5000.]), jnp.array([1000.])
        default = rayleigh.H2_rayleigh(wavelengths, x, temperatures)
        np.testing.assert_allclose(
            rayleigh.H2_rayleigh(wavelengths, x, temperatures, path_input_data=str(tmp_path)),
            2. * default, rtol=1e-12
        )
        np.testing.assert_allclose(
            rayleigh.compute_rayleigh_scattering("H2", x, wavelengths, x, temperatures, x,
                                                 path_input_data=str(tmp_path)),
            2. * default, rtol=1e-12
        )
        np.testing.assert_allclose(
            compute_rayleigh_scattering_opacities(
                ("H2",), jnp.ones(1), temperatures, {"H2": x}, x,
                jnp.array([cst.c / 5e-5]), path_input_data=str(tmp_path)
            ),
            2. * default, rtol=1e-6
        )
        # The default directory has not been displaced by the other one.
        np.testing.assert_array_equal(rayleigh.H2_rayleigh(wavelengths, x, temperatures), default)


@pytest.mark.parametrize("directory_exists", [False, True])
def test_missing_h2_table_is_downloaded_from_keeper(tmp_path, monkeypatch, directory_exists):
    requested_extensions = []

    def list_keeper_files(path, ext, **_):
        requested_extensions.append(ext)
        return {rayleigh._H2_TABLE_FILE: "https://keeper/file"}

    def download(destination, source, **_):
        assert source == "https://keeper/file"
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(table_file, destination)

    table_file = _h2_table_file()
    monkeypatch.setattr(_input_data, "_get_cli_functions", lambda: (download, list_keeper_files))
    if directory_exists:
        (tmp_path / get_input_data_subpaths()["rayleigh_opacities"]).mkdir(parents=True)
    file = rayleigh._get_h2_table_file(str(tmp_path))
    assert requested_extensions == ["h5"]
    assert file == os.path.join(str(tmp_path), get_input_data_subpaths()["rayleigh_opacities"],
                                rayleigh._H2_TABLE_FILE)
    assert os.path.isfile(file)


def test_h2_source_integrity():
    table = _full_h2_table()
    # Date_ID makes each written file unique, so the hash covers the source data rather than the file.
    assert hashlib.sha256(
        table["sigma_over_energy4_mb"].tobytes() + table["raman_over_energy4_mb"].tobytes()
    ).hexdigest() == H2_TABLE_SHA256
    assert len(table["rotational_n"]) == 293
    assert np.all(np.diff(table["photon_energy_ev"]) > 0)
    assert np.all(np.isfinite(table["sigma_over_energy4_mb"])) and np.all(table["sigma_over_energy4_mb"] > 0)
    assert np.all(np.isfinite(table["raman_over_energy4_mb"])) and np.all(table["raman_over_energy4_mb"] >= 0)


def test_h2_table_has_the_format_of_the_other_data_files():
    assert Opacity.split_species_all_info(rayleigh._H2_TABLE_FILE.split(".rayleightable")[0]) == (
        "1H2", "", "", "", "Singor", "DeltaWavenumber363_0.207-27.5mu"
    )
    assert rayleigh._H2_TABLE_FILE.endswith(".rayleightable.petitRADTRANS.h5")
    with h5py.File(_h2_table_file(), "r") as table:
        for name in ("DOI", "Date_ID", "mol_name", "mol_mass", "wavenumbers", "wlrange", "wnrange"):
            assert name in table
        for name, dataset in table.items():
            assert "long_name" in dataset.attrs, name
        for name in ("mol_mass", "wavenumbers", "wlrange", "wnrange", "excitation_energies",
                     "elastic_reduced_cross_sections", "raman_reduced_cross_sections"):
            assert "units" in table[name].attrs, name
        assert table["DOI"][0].decode() == "10.5281/zenodo.13441472"
        assert table["mol_name"][0].decode() == "H2"
        assert table["channel_count"][()] == 9582
        assert table["source_archive_md5"][0].decode() == "40b5f1679bdcfdac4dcde5cf0fb86870"
        np.testing.assert_allclose(table["wlrange"][:], [0.207, 27.5], atol=0.05)


def test_h2_reduced_rank_reproduces_full_thermal_sum():
    table = _full_h2_table()
    grid, n, excitation = table["photon_energy_ev"], table["rotational_n"], table["excitation_energy_ev"]
    total = rayleigh._H2_PECK_HUANG_CALIBRATION * (table["sigma_over_energy4_mb"] + table["raman_over_energy4_mb"])
    weights = (2 * n + 1) * np.where(n % 2 == 1, 3., 1.)
    wavelengths = rayleigh.HC_EV_UM / grid[1:-1] * 1e4
    # Dense in temperature, and deliberately off the tabulation nodes.
    temperatures = np.geomspace(1.13, 9500., 301)
    with jax.enable_x64(True):
        result = np.asarray(rayleigh.H2_rayleigh(jnp.asarray(wavelengths), jnp.ones(temperatures.size),
                                                 jnp.asarray(temperatures)))
    for j, temperature in enumerate(temperatures):
        log_weight = np.log(weights) - excitation / (rayleigh.KB_EV_K * temperature)
        populations = np.exp(log_weight - log_weight.max())
        populations /= populations.sum()
        reference = (populations @ total)[1:-1] * grid[1:-1]**4 * 1e-18 / (2 * cst.amu)
        np.testing.assert_allclose(result[:, j], reference, rtol=5e-4)


def test_h2_calibration_and_raman():
    table = _full_h2_table()
    grid, n, v = table["photon_energy_ev"], table["rotational_n"], table["vibrational_v"]
    ground = (n == 0) & (v == 0)
    with jax.enable_x64(True):
        for wavelength_um in (0.3, 0.5, 0.633, 1.0):
            energy = rayleigh.HC_EV_UM / wavelength_um
            # N=0 elastic scattering is purely isotropic, and lies just below the thermal mean of
            # 273 K normal hydrogen, which is what Peck & Huang (1977) measured.
            isotropic = (rayleigh._H2_PECK_HUANG_CALIBRATION * 1e-18 * energy**4
                         * np.interp(energy**2, grid**2, table["sigma_over_energy4_mb"][ground][0]))
            np.testing.assert_allclose(isotropic / peck_huang_sigma(wavelength_um), 0.995, atol=4e-3)
            # The total lies above the isotropic cross section by a King factor of a few per cent.
            total = kappa("H2", wavelength_um, 273.15)[0] * 2 * cst.amu
            assert 1.02 < total / peck_huang_sigma(wavelength_um) < 1.06


def test_h2_temperature_dependence_and_ortho_fraction():
    with jax.enable_x64(True):
        # Singor et al. (2024): Rayleigh at 3000 K is about 10% above 300 K at 2 eV. With Raman,
        # which rises with temperature, the total rises by somewhat more.
        ratio = kappa("H2", rayleigh.HC_EV_UM / 2., 3000.)[0] / kappa("H2", rayleigh.HC_EV_UM / 2., 300.)[0]
        assert 1.09 < ratio < 1.14
        wavelength, temperature = jnp.array([5000.]), jnp.array([300.])
        para = rayleigh.H2_rayleigh(wavelength, jnp.ones(1), temperature, 0.)
        ortho = rayleigh.H2_rayleigh(wavelength, jnp.ones(1), temperature, 1.)
        mixture = rayleigh.H2_rayleigh(wavelength, jnp.ones(1), temperature, 0.75)
        np.testing.assert_allclose(mixture, 0.25 * para + 0.75 * ortho, rtol=1e-12)
        # Thermal equilibrium at 300 K is close to the 3:1 ratio.
        np.testing.assert_allclose(rayleigh.H2_rayleigh(wavelength, jnp.ones(1), temperature), mixture, rtol=2e-3)


def test_unknown_species_is_rejected():
    with pytest.raises(ValueError, match="no Rayleigh scattering implementation"):
        rayleigh.compute_rayleigh_scattering("Xe", *inputs("He", jnp.float32))


@pytest.mark.parametrize("species", MASSES)
def test_jit_matches_eager_without_loading_tracers_into_cache(species):
    with jax.enable_x64(False):
        args = inputs(species, jnp.float32)
        compiled = jax.jit(lambda *args: rayleigh.compute_rayleigh_scattering(species, *args))
        np.testing.assert_allclose(compiled(*args), rayleigh.compute_rayleigh_scattering(species, *args),
                                   rtol=3e-6, atol=0.)
