"""Test petitRADTRANS chemistry module."""

import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import test_parameters, temperature_guillot_2010

relative_tolerance = 1e-6


def test_chemistry_atmosphere():
    c_o_ratios = test_parameters["chemical_parameters"]["c_o_ratios"][1] * np.ones_like(
        test_parameters["pressures"]
    )
    metallicities = test_parameters["chemical_parameters"]["metallicities"][
        1
    ] * np.ones_like(test_parameters["pressures"])

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperature_guillot_2010,
        pressures=test_parameters["pressures"],
        full=True,
    )


def test_chemistry_atmosphere_quench():
    c_o_ratios = test_parameters["chemical_parameters"]["c_o_ratios"][1] * np.ones_like(
        test_parameters["pressures"]
    )
    metallicities = test_parameters["chemical_parameters"]["metallicities"][
        1
    ] * np.ones_like(test_parameters["pressures"])

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperature_guillot_2010,
        pressures=test_parameters["pressures"],
        carbon_quench_pressure=test_parameters["chemical_parameters"][
            "pressure_quench_carbon"
        ],
        full=True,
    )


def test_chemistry_c_o_ratios():
    c_o_ratios = np.asarray(test_parameters["chemical_parameters"]["c_o_ratios"])
    metallicities = test_parameters["chemical_parameters"]["metallicities"][
        1
    ] * np.ones_like(c_o_ratios)
    pressures = test_parameters["chemical_parameters"]["pressure"] * np.ones_like(
        c_o_ratios
    )
    temperatures = test_parameters["chemical_parameters"]["temperature"] * np.ones_like(
        c_o_ratios
    )

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperatures,
        pressures=pressures,
        full=True,
    )


def test_chemistry_metallicities():
    metallicities = np.asarray(test_parameters["chemical_parameters"]["metallicities"])
    c_o_ratios = test_parameters["chemical_parameters"]["c_o_ratios"][1] * np.ones_like(
        metallicities
    )
    pressures = test_parameters["chemical_parameters"]["pressure"] * np.ones_like(
        metallicities
    )
    temperatures = test_parameters["chemical_parameters"]["temperature"] * np.ones_like(
        metallicities
    )

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperatures,
        pressures=pressures,
        full=True,
    )


def test_mass_fractions2metallicities():
    c_o_ratios = test_parameters["chemical_parameters"]["c_o_ratios"][1] * np.ones_like(
        test_parameters["pressures"]
    )
    metallicities = test_parameters["chemical_parameters"]["metallicities"][
        1
    ] * np.ones_like(test_parameters["pressures"])

    mass_fractions = petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperature_guillot_2010,
        pressures=test_parameters["pressures"],
        full=False,
    )

    sum_mass_fractions = np.sum(list(mass_fractions.values()), axis=0)

    mass_fractions = {
        species: mass_fraction / sum_mass_fractions
        for species, mass_fraction in mass_fractions.items()
    }

    mean_molar_masses = petitRADTRANS.chemistry.utils.compute_mean_molar_masses(
        abundances=mass_fractions
    )

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.utils.mass_fractions2metallicity,
        relative_tolerance=relative_tolerance,
    )

    benchmark.run(mass_fractions=mass_fractions, mean_molar_masses=mean_molar_masses)


def test_get_abundances_without_clouds_keeps_dict_p_bases_and_identity_indices(
    monkeypatch,
):
    pressures = np.array([1.0, 10.0])
    temperatures = np.array([1000.0, 1100.0])

    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        "calculate_gas_mass_fractions",
        lambda **kwargs: (
            {
                "FeH": np.array([0.2, 0.4]),
                "H2": np.array([0.7, 0.5]),
                "He": np.array([0.1, 0.1]),
            },
            1.0,
        ),
    )
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        "compute_mean_molar_masses",
        lambda abundances: np.ones_like(next(iter(abundances.values()))),
    )
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        "mass_fractions2metallicity",
        lambda mass_fractions, mean_molar_mass: (
            np.array([0.0, 0.0]),
            {
                6: {"local": np.array([0.5, 0.5])},
                8: {"local": np.array([1.0, 1.0])},
            },
        ),
    )

    abundances, mean_molar_mass, pressure_indices, p_bases = (
        petitRADTRANS.chemistry.core.get_abundances(
            pressures=pressures,
            temperatures=temperatures,
            parameters={
                "Fe/H": SimpleNamespace(value=0.0),
                "C/O": SimpleNamespace(value=0.5),
            },
            line_species=(),
            cloud_species=(),
        )
    )

    assert p_bases == {}
    np.testing.assert_array_equal(np.asarray(pressure_indices), np.array([0, 1]))
    np.testing.assert_allclose(np.asarray(abundances["FeH"]), np.array([0.1, 0.2]))
    np.testing.assert_allclose(np.asarray(mean_molar_mass), np.array([1.0, 1.0]))


def test_get_abundances_defaults_to_precalculated_equilibrium_table(
    monkeypatch,
):
    pressures = np.array([1.0, 10.0])
    temperatures = np.array([1000.0, 1100.0])
    captured = {}

    def fake_calculate_gas_mass_fractions(**kwargs):
        captured["use_equilibrium_chemistry"] = kwargs["use_equilibrium_chemistry"]
        return (
            {
                "H2": np.array([0.74, 0.74]),
                "He": np.array([0.26, 0.26]),
            },
            np.ones_like(pressures),
        )

    def fail_full_easychem(**_kwargs):
        raise AssertionError("full easyChem should not run without an explicit request")

    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        "calculate_gas_mass_fractions",
        fake_calculate_gas_mass_fractions,
    )
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        "get_easychem_abundances",
        fail_full_easychem,
    )
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        "compute_mean_molar_masses",
        lambda abundances: np.ones_like(next(iter(abundances.values()))),
    )

    abundances, mean_molar_mass, pressure_indices, p_bases = (
        petitRADTRANS.chemistry.core.get_abundances(
            pressures=pressures,
            temperatures=temperatures,
            parameters={
                "Fe/H": 0.0,
                "C/O": 0.55,
                "use_easychem": False,
            },
            line_species=(),
            cloud_species=(),
        )
    )

    assert captured["use_equilibrium_chemistry"] is True
    assert p_bases == {}
    np.testing.assert_array_equal(np.asarray(pressure_indices), np.array([0, 1]))
    np.testing.assert_allclose(np.asarray(abundances["H2"]), np.array([0.74, 0.74]))
    np.testing.assert_allclose(np.asarray(mean_molar_mass), np.array([1.0, 1.0]))


def test_get_abundances_uses_full_easychem_only_when_explicitly_requested(
    monkeypatch,
):
    pressures = np.array([1.0, 10.0])
    temperatures = np.array([1000.0, 1100.0])
    calls = {"easychem": 0}

    def fail_precalculated_table(**_kwargs):
        raise AssertionError(
            "the precalculated equilibrium table should not run when easyChem is requested"
        )

    def fake_easychem_abundances(**kwargs):
        calls["easychem"] += 1
        assert kwargs["metallicity"] == 0.0
        assert kwargs["carbon_to_oxygen"] == 0.55
        return {
            "FeH": np.array([0.2, 0.4]),
            "H2": np.array([0.7, 0.6]),
            "He": np.array([0.1, 0.2]),
            "MMW": np.array([2.0, 2.1]),
        }

    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        "calculate_gas_mass_fractions",
        fail_precalculated_table,
    )
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        "get_easychem_abundances",
        fake_easychem_abundances,
    )

    abundances, mean_molar_mass, pressure_indices, p_bases = (
        petitRADTRANS.chemistry.core.get_abundances(
            pressures=pressures,
            temperatures=temperatures,
            parameters={
                "Fe/H": 0.0,
                "C/O": 0.55,
                "use_easychem": True,
            },
            line_species=(),
            cloud_species=(),
        )
    )

    assert calls["easychem"] == 1
    assert p_bases == {}
    np.testing.assert_array_equal(np.asarray(pressure_indices), np.array([0, 1]))
    np.testing.assert_allclose(np.asarray(abundances["FeH"]), np.array([0.1, 0.2]))
    np.testing.assert_allclose(np.asarray(mean_molar_mass), np.array([2.0, 2.1]))


def test_profile_node_helpers_accept_plain_and_legacy_parameters():
    pressures = jnp.logspace(-6.0, 2.0, 8)
    parameters = {
        "H2O_linear_abundance_profile": True,
        "H2O_n_abundance_nodes": 3,
        "H2O_n_pressure_nodes": 1,
        "H2O_interpolation_mode": "relative",
        "H2O_pressure_node_0": 4.0,
        "H2O_abundance_node_0": -6.0,
        "H2O_abundance_node_1": -5.0,
        "H2O_abundance_node_2": -4.0,
    }
    legacy_parameters = {
        key: SimpleNamespace(value=value) for key, value in parameters.items()
    }

    pressure_nodes_plain = petitRADTRANS.chemistry.utils.define_pressure_node_list(
        pressures,
        "H2O",
        parameters,
    )
    abundance_nodes_plain = petitRADTRANS.chemistry.utils.define_abundance_node_list(
        "H2O",
        parameters,
    )
    pressure_nodes_legacy = petitRADTRANS.chemistry.utils.define_pressure_node_list(
        pressures,
        "H2O",
        legacy_parameters,
    )
    abundance_nodes_legacy = petitRADTRANS.chemistry.utils.define_abundance_node_list(
        "H2O",
        legacy_parameters,
    )

    np.testing.assert_allclose(
        np.asarray(pressure_nodes_plain), np.array([-6.0, -2.0, 2.0])
    )
    np.testing.assert_allclose(
        np.asarray(pressure_nodes_plain), np.asarray(pressure_nodes_legacy)
    )
    np.testing.assert_allclose(
        np.asarray(abundance_nodes_plain), np.asarray(abundance_nodes_legacy)
    )


def test_build_abundance_profile_supports_jit_and_gradients():
    pressures = jnp.logspace(-6.0, 2.0, 32)
    pressure_nodes = jnp.array([-6.0, -2.0, 2.0])
    abundance_nodes = jnp.array([-6.0, -5.0, -4.0])

    linear_profile = jax.jit(
        lambda pnodes, anodes: petitRADTRANS.chemistry.utils.build_abundance_profile(
            pressures,
            pnodes,
            anodes,
            profile_mode="linear",
        )
    )(pressure_nodes, abundance_nodes)
    stepped_profile = jax.jit(
        lambda pnodes, anodes: petitRADTRANS.chemistry.utils.build_abundance_profile(
            pressures,
            pnodes,
            anodes,
            profile_mode="stepped",
        )
    )(pressure_nodes, abundance_nodes)
    cubic_pressure_grad = jax.grad(
        lambda pnodes: jnp.sum(
            petitRADTRANS.chemistry.utils.build_abundance_profile(
                pressures,
                pnodes,
                abundance_nodes,
                profile_mode="cubic",
            )
        )
    )(pressure_nodes)

    assert linear_profile.shape == pressures.shape
    assert stepped_profile.shape == pressures.shape
    assert np.unique(np.round(np.log10(np.asarray(stepped_profile)), 6)).size == 3
    assert np.any(np.abs(np.asarray(cubic_pressure_grad)) > 0)


def test_get_abundances_supports_pressure_varying_profiles():
    pressures = jnp.logspace(-6.0, 2.0, 32)
    temperatures = 1200.0 * jnp.ones_like(pressures)
    parameters = {
        "H2O_linear_abundance_profile": True,
        "H2O_n_abundance_nodes": 3,
        "H2O_n_pressure_nodes": 1,
        "H2O_interpolation_mode": "relative",
        "H2O_pressure_node_0": 4.0,
        "H2O_abundance_node_0": -6.0,
        "H2O_abundance_node_1": -5.0,
        "H2O_abundance_node_2": -4.0,
        "CO": -5.0,
    }

    abundances, mean_molar_mass, pressure_indices, p_bases = (
        petitRADTRANS.chemistry.core.get_abundances(
            pressures=pressures,
            temperatures=temperatures,
            parameters=parameters,
            line_species=("H2O", "CO"),
            cloud_species=(),
        )
    )
    constant_abundances, _, _, _ = petitRADTRANS.chemistry.core.get_abundances(
        pressures=pressures,
        temperatures=temperatures,
        parameters={"H2O": -4.0, "CO": -5.0},
        line_species=("H2O", "CO"),
        cloud_species=(),
    )

    h2o_profile = np.asarray(abundances["H2O"])
    h2o_constant = np.asarray(constant_abundances["H2O"])

    assert p_bases == {}
    np.testing.assert_array_equal(
        np.asarray(pressure_indices), np.arange(pressures.size)
    )
    assert h2o_profile.shape == pressures.shape
    assert np.asarray(mean_molar_mass).shape == pressures.shape
    assert not np.allclose(h2o_profile, h2o_profile[0])
    assert np.allclose(h2o_constant, h2o_constant[0])


def test_calculate_gas_mass_fractions_supports_pressure_varying_profiles():
    pressures = jnp.logspace(-6.0, 2.0, 32)
    temperatures = 1200.0 * jnp.ones_like(pressures)

    mass_fractions, m_sum_total = (
        petitRADTRANS.chemistry.core.calculate_gas_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            imposed_mass_fractions={"CO": -5.0},
            line_species=("H2O", "CO"),
            cloud_species=(),
            filling_species=("H2", "He"),
            use_equilibrium_chemistry=False,
            H2O_linear_abundance_profile=True,
            H2O_n_abundance_nodes=3,
            H2O_n_pressure_nodes=1,
            H2O_interpolation_mode="relative",
            H2O_pressure_node_0=4.0,
            H2O_abundance_node_0=-6.0,
            H2O_abundance_node_1=-5.0,
            H2O_abundance_node_2=-4.0,
        )
    )

    h2o_profile = np.asarray(mass_fractions["H2O"])
    co_profile = np.asarray(mass_fractions["CO"])

    assert h2o_profile.shape == pressures.shape
    assert not np.allclose(h2o_profile, h2o_profile[0])
    assert np.allclose(co_profile, co_profile[0])
    np.testing.assert_allclose(np.asarray(m_sum_total), np.ones(pressures.size))


def test_calculate_gas_mass_fractions_equilibrium_tuple_fill_stays_finite():
    pressures = jnp.logspace(-6.0, 2.0, 8)
    temperatures = 1200.0 * jnp.ones_like(pressures)

    mass_fractions, m_sum_total = (
        petitRADTRANS.chemistry.core.calculate_gas_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            imposed_mass_fractions={},
            line_species=(),
            cloud_species=(),
            filling_species=("H2", "He"),
            metallicity=0.0,
            co_ratio=0.55,
        )
    )

    assert np.isfinite(np.asarray(m_sum_total)).all()
    np.testing.assert_allclose(np.asarray(m_sum_total), np.ones(pressures.size))
    assert np.isfinite(np.asarray(mass_fractions["H2"])).all()
    assert np.isfinite(np.asarray(mass_fractions["He"])).all()


def test_calculate_gas_mass_fractions_accepts_tuple_filling_species():
    pressures = np.logspace(-6.0, 2.0, 5)
    temperatures = np.full_like(pressures, 1000.0)

    mass_fractions, _ = petitRADTRANS.chemistry.core.calculate_gas_mass_fractions(
        pressures=pressures,
        temperatures=temperatures,
        imposed_mass_fractions={},
        line_species=(),
        cloud_species=(),
        filling_species=("H2", "He"),
        use_equilibrium_chemistry=False,
    )

    assert set(mass_fractions.keys()) == {"H2", "He"}
    assert np.asarray(mass_fractions["H2"]).shape == pressures.shape
    assert np.asarray(mass_fractions["He"]).shape == pressures.shape
    assert (
        np.asarray(mass_fractions["H2"]).dtype == np.asarray(mass_fractions["He"]).dtype
    )


def test_calculate_gas_mass_fractions_ignores_metallicity_without_equilibrium():
    pressures = np.asarray(test_parameters["pressures"])
    temperatures = np.full_like(
        pressures,
        test_parameters["temperature_guillot_2010_parameters"][
            "equilibrium_temperature"
        ],
    )
    imposed_mass_fractions = {
        species: mass_fraction
        for species, mass_fraction in test_parameters[
            "mass_fractions_correlated_k"
        ].items()
        if species not in test_parameters["filling_species"]
    }

    mass_fractions, m_sum_total = (
        petitRADTRANS.chemistry.core.calculate_gas_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            imposed_mass_fractions=imposed_mass_fractions,
            line_species=test_parameters["spectrum_parameters"][
                "line_species_correlated_k"
            ],
            cloud_species=(),
            filling_species=test_parameters["filling_species"],
            use_equilibrium_chemistry=False,
            metallicity=10
            ** test_parameters["chemical_parameters"]["metallicities"][1],
            co_ratio=test_parameters["chemical_parameters"]["c_o_ratios"][0],
        )
    )

    imposed_total = np.zeros(pressures.shape)

    for value in imposed_mass_fractions.values():
        imposed_total = imposed_total + np.broadcast_to(
            np.asarray(value), pressures.shape
        )

    fill_mass = 1.0 - imposed_total
    total_fill_weight = sum(test_parameters["filling_species"].values())

    np.testing.assert_allclose(np.asarray(m_sum_total), np.ones(pressures.size))

    for species, weight in test_parameters["filling_species"].items():
        expected = np.full(pressures.shape, fill_mass * weight / total_fill_weight)
        np.testing.assert_allclose(np.asarray(mass_fractions[species]), expected)


def test_calculate_gas_mass_fractions_equilibrium_with_imposed_profiles_stays_normalized():
    pressures = np.asarray(test_parameters["pressures"])
    temperatures = np.asarray(temperature_guillot_2010)
    line_species = test_parameters["spectrum_parameters"]["line_species_correlated_k"]
    imposed_mass_fractions = {
        species: np.logspace(
            test_parameters["mass_fractions_test"][0],
            test_parameters["mass_fractions_test"][1],
            pressures.size,
        )
        for species in line_species
    }

    mass_fractions, m_sum_total = (
        petitRADTRANS.chemistry.core.calculate_gas_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            imposed_mass_fractions=imposed_mass_fractions,
            line_species=line_species,
            cloud_species=(),
            filling_species=test_parameters["filling_species"],
            use_equilibrium_chemistry=True,
            metallicity=10
            ** test_parameters["chemical_parameters"]["metallicities"][1],
            co_ratio=test_parameters["chemical_parameters"]["c_o_ratios"][0],
        )
    )

    summed_mass_fractions = None
    for profile in mass_fractions.values():
        array = np.asarray(profile)
        summed_mass_fractions = (
            array.copy()
            if summed_mass_fractions is None
            else summed_mass_fractions + array
        )

    np.testing.assert_allclose(np.asarray(m_sum_total), np.ones(pressures.size))
    np.testing.assert_allclose(summed_mass_fractions, np.ones(pressures.size))
    assert np.all(np.asarray(mass_fractions["H2"]) >= 0)
    assert np.all(np.asarray(mass_fractions["He"]) >= 0)
    assert np.all(np.isfinite(np.asarray(mass_fractions["H2"])))
    assert np.all(np.isfinite(np.asarray(mass_fractions["He"])))


def test_calculate_gas_mass_fractions_builds_rare_isotopologue_profiles_from_ratios(
    monkeypatch,
):
    pressures = jnp.logspace(-6.0, 2.0, 6)
    temperatures = 1200.0 * jnp.ones_like(pressures)
    line_species = (
        "12C-16O__HITEMP.R1000_0.1-250mu",
        "13C-16O__HITEMP.R1000_0.1-250mu",
        "1H2-16O__HITEMP.R1000_0.1-250mu",
        "1H2-18O__HITEMP.R1000_0.1-250mu",
        "14N-1H3__HITRAN.R1000_0.1-250mu",
        "15N-1H3__HITRAN.R1000_0.1-250mu",
    )

    def fake_interpolate_mass_fractions(*_args, **_kwargs):
        chemistry_table = {
            "CO": 1e-4 * jnp.ones_like(pressures),
            "H2O": 2e-4 * jnp.ones_like(pressures),
            "NH3": 3e-5 * jnp.ones_like(pressures),
        }
        return chemistry_table, None, None

    monkeypatch.setattr(
        petitRADTRANS.chemistry.core.pre_calculated_equilibrium_chemistry_table,
        "interpolate_mass_fractions",
        fake_interpolate_mass_fractions,
    )

    ratio_parameters = {
        "13C/12C": 0.02,
        "18O/16O": 0.01,
        "15N/14N": 0.03,
    }

    mass_fractions_with_ratios, m_sum_total = (
        petitRADTRANS.chemistry.core.calculate_gas_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            imposed_mass_fractions={},
            line_species=line_species,
            cloud_species=(),
            filling_species={"H2": 74.0, "He": 24.0},
            use_equilibrium_chemistry=True,
            metallicity=0.0,
            co_ratio=0.55,
            parameters=ratio_parameters,
        )
    )

    mass_fractions_without_ratios, _ = (
        petitRADTRANS.chemistry.core.calculate_gas_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            imposed_mass_fractions={},
            line_species=line_species,
            cloud_species=(),
            filling_species={"H2": 74.0, "He": 24.0},
            use_equilibrium_chemistry=True,
            metallicity=0.0,
            co_ratio=0.55,
            parameters={},
        )
    )

    co_main = np.asarray(mass_fractions_with_ratios["12C-16O__HITEMP.R1000_0.1-250mu"])
    h2o_main = np.asarray(mass_fractions_with_ratios["1H2-16O__HITEMP.R1000_0.1-250mu"])
    nh3_main = np.asarray(mass_fractions_with_ratios["14N-1H3__HITRAN.R1000_0.1-250mu"])

    np.testing.assert_allclose(
        np.asarray(mass_fractions_with_ratios["13C-16O__HITEMP.R1000_0.1-250mu"]),
        0.02 * co_main,
    )
    np.testing.assert_allclose(
        np.asarray(mass_fractions_with_ratios["1H2-18O__HITEMP.R1000_0.1-250mu"]),
        0.01 * h2o_main,
    )
    np.testing.assert_allclose(
        np.asarray(mass_fractions_with_ratios["15N-1H3__HITRAN.R1000_0.1-250mu"]),
        0.03 * nh3_main,
    )
    np.testing.assert_allclose(np.asarray(m_sum_total), np.ones(pressures.size))

    assert np.all(
        np.asarray(mass_fractions_with_ratios["H2"])
        < np.asarray(mass_fractions_without_ratios["H2"])
    )


def test_calculate_cloud_mass_fractions_uses_free_cloud_base_without_equilibrium_parameters(
    monkeypatch,
):
    pressures = jnp.array([1.0, 10.0])
    temperatures = jnp.array([1000.0, 1100.0])
    mean_molar_mass = jnp.array([2.3, 2.3])
    calls = {"eq": 0, "free": 0}

    class FakeCloudOpacity:
        def __init__(self, *_args, **_kwargs):
            self.species_full_name = "StubCloud_placeholder"

    def fake_eq_case(**_kwargs):
        calls["eq"] += 1
        return jnp.array(11.0)

    def fake_free_case(*_args, **_kwargs):
        calls["free"] += 1
        return jnp.array(22.0)

    monkeypatch.setattr(petitRADTRANS.chemistry.core, "CloudOpacity", FakeCloudOpacity)
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core.fc,
        "calculate_cloud_base_pressure",
        fake_eq_case,
    )
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core.fc,
        "calculate_cloud_base_pressure_free",
        fake_free_case,
    )

    mass_fractions, p_bases, pressure_indices = (
        petitRADTRANS.chemistry.core.calculate_cloud_mass_fractions(
            pressures=pressures,
            temperatures=temperatures,
            metallicity=0.0,
            co_ratio=0.55,
            cloud_species=("dummy",),
            parameters={
                "log_X_cb_StubCloud": -5.0,
                "fsed": 2.0,
            },
            mass_fractions={
                "H2": jnp.array([0.75, 0.75]),
                "He": jnp.array([0.25, 0.25]),
            },
            mean_molar_mass=mean_molar_mass,
            adaptive_mesh_refinement=False,
            use_easychem=False,
        )
    )

    assert calls == {"eq": 0, "free": 1}
    np.testing.assert_allclose(np.asarray(p_bases["StubCloud"]), np.array(22.0))
    np.testing.assert_array_equal(np.asarray(pressure_indices), np.array([0, 1]))
    assert np.asarray(mass_fractions["dummy"]).shape == pressures.shape
