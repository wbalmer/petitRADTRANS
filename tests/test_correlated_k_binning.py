"""Test the JAX re-implementation of the exo-k correlated-k binning (petitRADTRANS.opacities.correlated_k_binning).

The k-tables are small synthetic tables written to a temporary directory. Where exo_k is installed, the binned
tables are also compared with the ones of the original exo-k implementation.
"""
from jax import config

config.update("jax_enable_x64", True)

import os  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from .context import petitRADTRANS  # noqa: E402,F401

import petitRADTRANS.opacities.correlated_k_binning as correlated_k_binning  # noqa: E402
import petitRADTRANS.opacities.file_conversion as file_conversion  # noqa: E402

Ktable = correlated_k_binning.Ktable

_N_PRESSURES = 3
_N_TEMPERATURES = 2
_N_WAVENUMBERS = 24
_WAVENUMBER_EDGES = np.linspace(8000.0, 11000.0, _N_WAVENUMBERS + 1)  # (cm-1)


def _gauss_legendre_unit_interval(n_points):
    samples, weights = np.polynomial.legendre.leggauss(n_points)

    return 0.5 * (samples + 1.0), 0.5 * weights


_G_SAMPLES, _G_WEIGHTS = _gauss_legendre_unit_interval(8)


def _synthetic_kcoeff(seed=0, n_g=_G_SAMPLES.size):
    """Monotonically increasing k-distributions spanning several orders of magnitude (cm2/molecule)."""
    rng = np.random.default_rng(seed)
    log10_k = (
        -24.0
        + rng.normal(0.0, 1.0, (_N_PRESSURES, _N_TEMPERATURES, _N_WAVENUMBERS, 1))
        + np.sort(rng.uniform(0.0, 4.0, (_N_PRESSURES, _N_TEMPERATURES, _N_WAVENUMBERS, n_g)), axis=-1)
    )

    return 10.0 ** log10_k


def _write_ktable(
    file,
    kcoeff,
    wavenumber_edges=_WAVENUMBER_EDGES,
    samples=_G_SAMPLES,
    weights=_G_WEIGHTS,
    mol_name="H2O",
    mol_mass=18.0,
    kcoeff_unit="cm^2/molecule",
):
    """Write a k-table in the format of the petitRADTRANS (ExoMol) k-tables."""
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(file, "w") as f:
        f.create_dataset("DOI", (1,), data="test DOI", dtype=string_dtype)
        f.create_dataset("method", (1,), data="petit_samples", dtype=string_dtype)

        if mol_name is not None:
            f.create_dataset("mol_name", (1,), data=mol_name, dtype=string_dtype)

        if mol_mass is not None:
            f.create_dataset("mol_mass", (1,), data=mol_mass)
            f["mol_mass"].attrs["units"] = "AMU"

        f.create_dataset("bin_edges", data=wavenumber_edges)
        f["bin_edges"].attrs["units"] = "cm^-1"
        f.create_dataset("bin_centers", data=0.5 * (wavenumber_edges[1:] + wavenumber_edges[:-1]))
        f["bin_centers"].attrs["units"] = "cm^-1"
        f.create_dataset("kcoeff", data=kcoeff)
        f["kcoeff"].attrs["units"] = kcoeff_unit
        f.create_dataset("p", data=np.logspace(-3.0, 1.0, kcoeff.shape[0]))
        f["p"].attrs["units"] = "bar"
        f.create_dataset("t", data=np.linspace(500.0, 1500.0, kcoeff.shape[1]))
        f["t"].attrs["units"] = "K"
        f.create_dataset("samples", data=samples)
        f.create_dataset("weights", data=weights)

    return file


@pytest.fixture
def ktable_file(tmp_path):
    return _write_ktable(os.path.join(tmp_path, "H2O_test.ktable.h5"), _synthetic_kcoeff())


def test_rebin_averages_over_the_overlap_of_the_bins():
    fine_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    coarse_grid = np.array([0.5, 2.0, 4.0, 5.0, 6.0])

    rebinned = correlated_k_binning.rebin(values, fine_grid, coarse_grid)

    # [0.5, 2]: 0.5 * 1 + 1 * 2 over 1.5; [2, 4]: mean of 3 and 4; [4, 5] and [5, 6]: no overlap
    np.testing.assert_allclose(rebinned, [2.5 / 1.5, 3.5, 0.0, 0.0], rtol=1e-14)


def test_rebin_ind_weights_reproduces_the_bin_averages():
    old_edges = np.array([0.0, 1.0, 2.5, 3.0, 5.0, 6.0])
    new_edges = np.array([0.5, 0.75, 2.75, 5.5])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    indices, weights, weight_counts = correlated_k_binning.rebin_ind_weights(old_edges, new_edges)

    assert weights.shape == (new_edges.size - 1, np.max(weight_counts))
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0, rtol=1e-14)

    # Formula of the rebin_ind_weights docstring
    averages = [
        np.dot(values[indices[i] - 1: indices[i] - 1 + weight_counts[i]], weights[i, : weight_counts[i]])
        for i in range(new_edges.size - 1)
    ]
    expected = correlated_k_binning.rebin(values, old_edges, new_edges)

    np.testing.assert_allclose(averages, expected, rtol=1e-14)
    # A new bin inside a single old bin takes the value of that bin
    np.testing.assert_allclose(averages[0], 1.0, rtol=1e-14)


def test_is_sorted():
    assert bool(correlated_k_binning.is_sorted(np.array([1.0, 2.0, 2.0, 3.0])))
    assert not bool(correlated_k_binning.is_sorted(np.array([1.0, 3.0, 2.0])))


def test_rm_molec():
    assert correlated_k_binning.rm_molec("cm^2/molecule") == "cm^2"
    assert correlated_k_binning.rm_molec("cm^2/molec") == "cm^2"
    assert correlated_k_binning.rm_molec("cm^2") == "cm^2"


def test_unit_convert():
    assert correlated_k_binning.unit_convert("p_unit", unit_file="bar") == ("bar", 1.0)

    unit, conversion_factor = correlated_k_binning.unit_convert("p_unit", unit_file="bar", unit_out="Pa")
    assert unit == "Pa"
    np.testing.assert_allclose(conversion_factor, 1e5, rtol=1e-14)

    # The assumed unit takes precedence over the unit of the file
    unit, conversion_factor = correlated_k_binning.unit_convert(
        "p_unit", unit_file="bar", unit_in="mbar", unit_out="bar"
    )
    assert unit == "bar"
    np.testing.assert_allclose(conversion_factor, 1e-3, rtol=1e-14)

    with pytest.raises(NotImplementedError):
        correlated_k_binning.unit_convert("p_unit")


def test_select_spectral_range():
    def spectral_object():
        _spectral_object = correlated_k_binning.Spectral_object()
        _spectral_object.wnedges = _WAVENUMBER_EDGES.copy()
        _spectral_object.wns = 0.5 * (_WAVENUMBER_EDGES[1:] + _WAVENUMBER_EDGES[:-1])

        return _spectral_object

    assert spectral_object().select_spectral_range() == (None, None)

    # Only the bins entirely inside the range are selected
    by_wavenumber = spectral_object()
    i_min, i_max = by_wavenumber.select_spectral_range(wn_range=[8400.0, 9100.0])
    np.testing.assert_allclose(by_wavenumber.wnedges[[0, -1]], [8500.0, 9000.0])
    assert by_wavenumber.Nw == by_wavenumber.wns.size == i_max - i_min == 4

    by_wavelength = spectral_object()
    by_wavelength.select_spectral_range(wl_range=[1e4 / 9100.0, 1e4 / 8400.0])
    np.testing.assert_allclose(by_wavelength.wnedges, by_wavenumber.wnedges)
    np.testing.assert_allclose(by_wavelength.wlrange, 1e4 / by_wavenumber.wnedges[[-1, 0]])

    with pytest.raises(RuntimeError, match="either wn_range or wl_range"):
        spectral_object().select_spectral_range(wn_range=[8400.0, 9100.0], wl_range=[1.0, 1.1])

    with pytest.raises(RuntimeError, match="does not contain any point"):
        spectral_object().select_spectral_range(wn_range=[8510.0, 8520.0])


def test_ktable_reads_hdf5_file(ktable_file):
    kcoeff = _synthetic_kcoeff()
    ktable = Ktable(filename=ktable_file)

    assert ktable.mol == "H2O"
    assert ktable.DOI == "test DOI"
    assert ktable.sampling_method == "petit_samples"
    assert (ktable.Np, ktable.Nt, ktable.Nw, ktable.Ng) == kcoeff.shape
    assert ktable.kdata_unit == "cm^2/molecule"
    assert ktable.p_unit == "bar"
    np.testing.assert_allclose(ktable.kdata, kcoeff, rtol=0)
    np.testing.assert_allclose(ktable.gedges, np.concatenate(([0.0], np.cumsum(_G_WEIGHTS))), rtol=1e-14)
    np.testing.assert_allclose(ktable.wnrange, _WAVENUMBER_EDGES[[0, -1]])
    np.testing.assert_allclose(ktable.molar_mass, 18e-3, rtol=1e-14)  # kg/mol


def test_ktable_falls_back_to_the_file_name_and_formula_for_the_molecule(tmp_path):
    file = _write_ktable(
        os.path.join(tmp_path, "CO2_test.ktable.h5"), _synthetic_kcoeff(), mol_name=None, mol_mass=None
    )
    ktable = Ktable(filename=file)

    assert ktable.mol == "CO2"
    np.testing.assert_allclose(ktable.molar_mass, 44e-3, rtol=1e-3)
    assert Ktable(filename=file, mol="CO").mol == "CO"


def test_ktable_rejects_files_without_weights_or_hdf5_extension(tmp_path, ktable_file):
    with pytest.raises(RuntimeError, match="hdf5"):
        Ktable(filename=os.path.join(tmp_path, "H2O_test.ktable.dat"))

    with h5py.File(ktable_file, "a") as f:
        del f["weights"]

    with pytest.raises(RuntimeError, match="No weights"):
        Ktable(filename=ktable_file)


def test_ktable_converts_units(ktable_file):
    kcoeff = _synthetic_kcoeff()
    ktable = Ktable(filename=ktable_file, p_unit="Pa", kdata_unit="m^2")

    assert ktable.p_unit == "Pa"
    assert ktable.kdata_unit == "m^2/molecule"
    np.testing.assert_allclose(ktable.pgrid, np.logspace(-3.0, 1.0, _N_PRESSURES) * 1e5, rtol=1e-14)
    np.testing.assert_allclose(ktable.logpgrid, np.log10(ktable.pgrid), rtol=1e-14)
    np.testing.assert_allclose(ktable.kdata, kcoeff * 1e-4, rtol=1e-14)


def test_ktable_removes_zeros(tmp_path):
    kcoeff = _synthetic_kcoeff()
    kcoeff[0, 0, 0, :] = 0.0
    file = _write_ktable(os.path.join(tmp_path, "H2O_zeros.ktable.h5"), kcoeff)

    np.testing.assert_array_equal(Ktable(filename=file, remove_zeros=False).kdata[0, 0, 0], 0.0)

    ktable = Ktable(filename=file, remove_zeros=True)

    np.testing.assert_allclose(ktable.kdata[0, 0, 0], np.min(kcoeff[kcoeff > 0]) * 1e-10, rtol=1e-14)
    np.testing.assert_array_equal(ktable.kdata[0, 0, 1:], kcoeff[0, 0, 1:])


def test_bin_down_of_identical_bins_does_not_depend_on_the_number_of_bins(tmp_path):
    # Mixing a k-distribution with itself must give the same result whatever the number of mixed bins
    kcoeff = np.broadcast_to(_synthetic_kcoeff()[:, :, :1, :], _synthetic_kcoeff().shape).copy()
    file = _write_ktable(os.path.join(tmp_path, "H2O_identical.ktable.h5"), kcoeff)

    mixed = Ktable(filename=file)
    mixed.bin_down(_WAVENUMBER_EDGES[::4])
    single = Ktable(filename=file)
    single.bin_down(_WAVENUMBER_EDGES)

    assert mixed.kdata.shape == (_N_PRESSURES, _N_TEMPERATURES, _N_WAVENUMBERS // 4, _G_SAMPLES.size)
    np.testing.assert_allclose(
        mixed.kdata, np.broadcast_to(single.kdata[:, :, :1, :], mixed.kdata.shape), rtol=1e-12
    )

    # The method sets the cumulative distribution to 0 and 1 at the first and last g-points, which shifts the
    # k-coefficients of these points (by ~10% with 8 g-points); the other ones are close to the original ones
    np.testing.assert_allclose(mixed.kdata[..., 1:-1], kcoeff[:, :, ::4, 1:-1], rtol=2e-2)


def test_bin_down_of_constant_k_coefficients_is_exact(tmp_path):
    kcoeff = np.full((_N_PRESSURES, _N_TEMPERATURES, _N_WAVENUMBERS, _G_SAMPLES.size), 3e-22)
    ktable = Ktable(filename=_write_ktable(os.path.join(tmp_path, "H2O_constant.ktable.h5"), kcoeff))

    ktable.bin_down(_WAVENUMBER_EDGES[::3])

    np.testing.assert_allclose(ktable.kdata, 3e-22, rtol=1e-14)


def test_bin_down_updates_the_spectral_and_g_grids(ktable_file):
    ktable = Ktable(filename=ktable_file)
    new_g_samples, new_g_weights = _gauss_legendre_unit_interval(4)
    new_edges = np.linspace(8500.0, 10500.0, 6)

    ktable.bin_down(new_edges, weights=new_g_weights, ggrid=new_g_samples)

    assert ktable.kdata.shape == (_N_PRESSURES, _N_TEMPERATURES, new_edges.size - 1, new_g_samples.size)
    assert (ktable.Nw, ktable.Ng) == (new_edges.size - 1, new_g_samples.size)
    np.testing.assert_allclose(ktable.wnedges, new_edges)
    np.testing.assert_allclose(ktable.wns, 0.5 * (new_edges[1:] + new_edges[:-1]))
    np.testing.assert_allclose(ktable.ggrid, new_g_samples)
    np.testing.assert_allclose(ktable.gedges, np.concatenate(([0.0], np.cumsum(new_g_weights))), rtol=1e-14)
    # The k-distributions remain sorted, and within the range of the original ones
    assert np.all(np.diff(ktable.kdata, axis=-1) >= 0)
    assert np.min(ktable.kdata) >= np.min(_synthetic_kcoeff()) * (1 - 1e-12)
    assert np.max(ktable.kdata) <= np.max(_synthetic_kcoeff()) * (1 + 1e-12)

    # Without ggrid, the g-points are at the middle of the bins defined by the weights
    ktable = Ktable(filename=ktable_file)
    ktable.bin_down(new_edges, weights=new_g_weights)

    gedges = np.concatenate(([0.0], np.cumsum(new_g_weights)))
    np.testing.assert_allclose(ktable.ggrid, 0.5 * (gedges[1:] + gedges[:-1]), rtol=1e-14)


def test_bin_down_leaves_bins_outside_of_the_table_empty(ktable_file):
    ktable = Ktable(filename=ktable_file)
    new_edges = np.array([7000.0, 7500.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0])

    ktable.bin_down(new_edges)

    # The first and last bins are outside the table; the bin [7500, 8000] has no data either
    np.testing.assert_array_equal(ktable.kdata[:, :, :2], 0.0)
    np.testing.assert_array_equal(ktable.kdata[:, :, -1], 0.0)
    assert np.all(ktable.kdata[:, :, 2:-1] > 0)


def test_bin_down_rejects_invalid_wavenumber_edges(ktable_file):
    with pytest.raises(TypeError, match="at least have two values"):
        Ktable(filename=ktable_file).bin_down([9000.0])

    with pytest.raises(RuntimeError, match="sorted"):
        Ktable(filename=ktable_file).bin_down([9000.0, 8500.0, 10000.0])


@pytest.mark.parametrize("use_rebin", [False, True])
def test_bin_down_matches_exo_k(ktable_file, use_rebin):
    exo_k = pytest.importorskip("exo_k")
    new_edges = np.logspace(np.log10(8100.0), np.log10(10900.0), 7)

    ktable = Ktable(filename=ktable_file, remove_zeros=True)
    ktable.bin_down(new_edges, use_rebin=use_rebin)

    reference = exo_k.Ktable(filename=ktable_file, remove_zeros=True)
    reference.bin_down(new_edges, use_rebin=use_rebin)

    np.testing.assert_allclose(ktable.kdata, reference.kdata, rtol=1e-10)
    np.testing.assert_allclose(ktable.wns, reference.wns, rtol=1e-14)


def test_write_hdf5_round_trip(tmp_path, ktable_file):
    ktable = Ktable(filename=ktable_file)
    ktable.bin_down(_WAVENUMBER_EDGES[::2])
    output_file = os.path.join(tmp_path, "binned", "H2O_binned.ktable.h5")

    ktable.write_hdf5(output_file)
    reloaded = Ktable(filename=output_file)

    np.testing.assert_allclose(reloaded.kdata, ktable.kdata, rtol=0)
    np.testing.assert_allclose(reloaded.wnedges, ktable.wnedges, rtol=0)
    np.testing.assert_allclose(reloaded.weights, ktable.weights, rtol=0)
    np.testing.assert_allclose(reloaded.ggrid, ktable.ggrid, rtol=0)
    np.testing.assert_allclose(reloaded.pgrid, ktable.pgrid, rtol=0)
    np.testing.assert_allclose(reloaded.tgrid, ktable.tgrid, rtol=0)
    assert reloaded.mol == "H2O"
    assert reloaded.DOI == "test DOI"

    with h5py.File(output_file, "r") as f:
        np.testing.assert_allclose(f["mol_mass"][()], [18.0], rtol=1e-14)
        assert f["mol_mass"].attrs["units"] == "AMU"
        np.testing.assert_allclose(f["wnrange"][()], _WAVENUMBER_EDGES[[0, -1]])
        np.testing.assert_allclose(f["wlrange"][()], 1e4 / _WAVENUMBER_EDGES[[-1, 0]])
        assert f["ngauss"][()] == _G_SAMPLES.size

    # The extension is added if missing
    ktable.write_hdf5(os.path.join(tmp_path, "H2O_no_extension"))
    assert os.path.isfile(os.path.join(tmp_path, "H2O_no_extension.h5"))


def test_write_hdf5_converts_back_to_exomol_units(tmp_path, ktable_file):
    ktable = Ktable(filename=ktable_file, p_unit="Pa", kdata_unit="m^2")
    output_file = os.path.join(tmp_path, "H2O_exomol_units.ktable.h5")

    ktable.write_hdf5(output_file, exomol_units=True)

    with h5py.File(output_file, "r") as f:
        assert f["kcoeff"].attrs["units"] == "cm^2/molecule"
        assert f["p"].attrs["units"] == "bar"
        np.testing.assert_allclose(f["kcoeff"][()], _synthetic_kcoeff(), rtol=1e-14)
        np.testing.assert_allclose(f["p"][()], np.logspace(-3.0, 1.0, _N_PRESSURES), rtol=1e-14)


def test_rebin_ck_line_opacities_writes_a_binned_down_file(tmp_path):
    input_file = _write_ktable(
        os.path.join(tmp_path, "1H2-16O__Test.R1000_0.9-1.25mu.ktable.petitRADTRANS.h5"), _synthetic_kcoeff()
    )
    new_edges = _WAVENUMBER_EDGES[::4]

    assert file_conversion.rebin_ck_line_opacities(
        input_file=input_file, target_resolving_power=10, wavenumber_grid=new_edges, disable_mpi=True
    ) == 0

    output_file = os.path.join(tmp_path, "1H2-16O__Test.R10_0.9-1.25mu.ktable.petitRADTRANS.h5")
    binned = Ktable(filename=output_file)
    expected = Ktable(filename=input_file, remove_zeros=True)
    expected.bin_down(new_edges)

    np.testing.assert_allclose(binned.kdata, expected.kdata, rtol=0)
