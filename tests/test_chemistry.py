"""Test petitRADTRANS chemistry module.
"""
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import test_parameters, temperature_guillot_2010

relative_tolerance = 1e-6


def test_chemistry_atmosphere():
    c_o_ratios = (
        test_parameters['chemical_parameters']['c_o_ratios'][1]
        * np.ones_like(test_parameters['pressures'])
    )
    metallicities = (
        test_parameters['chemical_parameters']['metallicities'][1]
        * np.ones_like(test_parameters['pressures'])
    )

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperature_guillot_2010,
        pressures=test_parameters['pressures'],
        full=True
    )


def test_chemistry_atmosphere_quench():
    c_o_ratios = test_parameters['chemical_parameters']['c_o_ratios'][1] \
                 * np.ones_like(test_parameters['pressures'])
    metallicities = test_parameters['chemical_parameters']['metallicities'][1] \
                    * np.ones_like(test_parameters['pressures'])

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperature_guillot_2010,
        pressures=test_parameters['pressures'],
        carbon_quench_pressure=test_parameters['chemical_parameters']['pressure_quench_carbon'],
        full=True
    )


def test_chemistry_c_o_ratios():
    c_o_ratios = np.asarray(test_parameters['chemical_parameters']['c_o_ratios'])
    metallicities = test_parameters['chemical_parameters']['metallicities'][1] * np.ones_like(c_o_ratios)
    pressures = test_parameters['chemical_parameters']['pressure'] * np.ones_like(c_o_ratios)
    temperatures = test_parameters['chemical_parameters']['temperature'] * np.ones_like(c_o_ratios)

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperatures,
        pressures=pressures,
        full=True
    )


def test_chemistry_metallicities():
    metallicities = np.asarray(test_parameters['chemical_parameters']['metallicities'])
    c_o_ratios = test_parameters['chemical_parameters']['c_o_ratios'][1] * np.ones_like(metallicities)
    pressures = test_parameters['chemical_parameters']['pressure'] * np.ones_like(metallicities)
    temperatures = test_parameters['chemical_parameters']['temperature'] * np.ones_like(metallicities)

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        co_ratios=c_o_ratios,
        log10_metallicities=metallicities,
        temperatures=temperatures,
        pressures=pressures,
        full=True
    )


def test_mass_fractions2metallicities():
    c_o_ratios = (
        test_parameters['chemical_parameters']['c_o_ratios'][1]
        * np.ones_like(test_parameters['pressures'])
    )
    metallicities = (
        test_parameters['chemical_parameters']['metallicities'][1]
        * np.ones_like(test_parameters['pressures'])
    )

    mass_fractions = (
        petitRADTRANS.chemistry.pre_calculated_chemistry.pre_calculated_equilibrium_chemistry_table.
        interpolate_mass_fractions(
            co_ratios=c_o_ratios,
            log10_metallicities=metallicities,
            temperatures=temperature_guillot_2010,
            pressures=test_parameters['pressures'],
            full=False
        )
    )

    sum_mass_fractions = np.sum(list(mass_fractions.values()), axis=0)

    mass_fractions = {species: mass_fraction / sum_mass_fractions for species, mass_fraction in mass_fractions.items()}

    mean_molar_masses = petitRADTRANS.chemistry.utils.compute_mean_molar_masses(
        abundances=mass_fractions
    )

    benchmark = Benchmark(
        function=petitRADTRANS.chemistry.utils.mass_fractions2metallicity,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses
    )


def test_get_abundances_without_clouds_keeps_dict_p_bases_and_identity_indices(monkeypatch):
    pressures = np.array([1.0, 10.0])
    temperatures = np.array([1000.0, 1100.0])

    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        'calculate_gas_mass_fractions',
        lambda **kwargs: (
            {
                'FeH': np.array([0.2, 0.4]),
                'H2': np.array([0.7, 0.5]),
                'He': np.array([0.1, 0.1]),
            },
            1.0,
        ),
    )
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        'compute_mean_molar_masses',
        lambda abundances: np.ones_like(next(iter(abundances.values()))),
    )
    monkeypatch.setattr(
        petitRADTRANS.chemistry.core,
        'mass_fractions2metallicity',
        lambda mass_fractions, mean_molar_mass: (
            np.array([0.0, 0.0]),
            {
                6: {'local': np.array([0.5, 0.5])},
                8: {'local': np.array([1.0, 1.0])},
            },
        ),
    )

    abundances, mean_molar_mass, pressure_indices, p_bases = petitRADTRANS.chemistry.core.get_abundances(
        pressures=pressures,
        temperatures=temperatures,
        parameters={
            'Fe/H': SimpleNamespace(value=0.0),
            'C/O': SimpleNamespace(value=0.5),
        },
        line_species=(),
        cloud_species=(),
    )

    assert p_bases == {}
    np.testing.assert_array_equal(np.asarray(pressure_indices), np.array([0, 1]))
    np.testing.assert_allclose(np.asarray(abundances['FeH']), np.array([0.1, 0.2]))
    np.testing.assert_allclose(np.asarray(mean_molar_mass), np.array([1.0, 1.0]))


def test_profile_node_helpers_accept_plain_and_legacy_parameters():
    pressures = jnp.logspace(-6.0, 2.0, 8)
    parameters = {
        'H2O_linear_abundance_profile': True,
        'H2O_n_abundance_nodes': 3,
        'H2O_n_pressure_nodes': 1,
        'H2O_interpolation_mode': 'relative',
        'H2O_pressure_node_0': 4.0,
        'H2O_abundance_node_0': -6.0,
        'H2O_abundance_node_1': -5.0,
        'H2O_abundance_node_2': -4.0,
    }
    legacy_parameters = {
        key: SimpleNamespace(value=value) for key, value in parameters.items()
    }

    pressure_nodes_plain = petitRADTRANS.chemistry.utils.define_pressure_node_list(
        pressures,
        'H2O',
        parameters,
    )
    abundance_nodes_plain = petitRADTRANS.chemistry.utils.define_abundance_node_list(
        'H2O',
        parameters,
    )
    pressure_nodes_legacy = petitRADTRANS.chemistry.utils.define_pressure_node_list(
        pressures,
        'H2O',
        legacy_parameters,
    )
    abundance_nodes_legacy = petitRADTRANS.chemistry.utils.define_abundance_node_list(
        'H2O',
        legacy_parameters,
    )

    np.testing.assert_allclose(np.asarray(pressure_nodes_plain), np.array([-6.0, -2.0, 2.0]))
    np.testing.assert_allclose(np.asarray(pressure_nodes_plain), np.asarray(pressure_nodes_legacy))
    np.testing.assert_allclose(np.asarray(abundance_nodes_plain), np.asarray(abundance_nodes_legacy))


def test_build_abundance_profile_supports_jit_and_gradients():
    pressures = jnp.logspace(-6.0, 2.0, 32)
    pressure_nodes = jnp.array([-6.0, -2.0, 2.0])
    abundance_nodes = jnp.array([-6.0, -5.0, -4.0])

    linear_profile = jax.jit(
        lambda pnodes, anodes: petitRADTRANS.chemistry.utils.build_abundance_profile(
            pressures,
            pnodes,
            anodes,
            profile_mode='linear',
        )
    )(pressure_nodes, abundance_nodes)
    stepped_profile = jax.jit(
        lambda pnodes, anodes: petitRADTRANS.chemistry.utils.build_abundance_profile(
            pressures,
            pnodes,
            anodes,
            profile_mode='stepped',
        )
    )(pressure_nodes, abundance_nodes)
    cubic_pressure_grad = jax.grad(
        lambda pnodes: jnp.sum(
            petitRADTRANS.chemistry.utils.build_abundance_profile(
                pressures,
                pnodes,
                abundance_nodes,
                profile_mode='cubic',
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
        'H2O_linear_abundance_profile': True,
        'H2O_n_abundance_nodes': 3,
        'H2O_n_pressure_nodes': 1,
        'H2O_interpolation_mode': 'relative',
        'H2O_pressure_node_0': 4.0,
        'H2O_abundance_node_0': -6.0,
        'H2O_abundance_node_1': -5.0,
        'H2O_abundance_node_2': -4.0,
        'CO': -5.0,
    }

    abundances, mean_molar_mass, pressure_indices, p_bases = petitRADTRANS.chemistry.core.get_abundances(
        pressures=pressures,
        temperatures=temperatures,
        parameters=parameters,
        line_species=('H2O', 'CO'),
        cloud_species=(),
    )
    constant_abundances, _, _, _ = petitRADTRANS.chemistry.core.get_abundances(
        pressures=pressures,
        temperatures=temperatures,
        parameters={'H2O': -4.0, 'CO': -5.0},
        line_species=('H2O', 'CO'),
        cloud_species=(),
    )

    h2o_profile = np.asarray(abundances['H2O'])
    h2o_constant = np.asarray(constant_abundances['H2O'])

    assert p_bases == {}
    np.testing.assert_array_equal(np.asarray(pressure_indices), np.arange(pressures.size))
    assert h2o_profile.shape == pressures.shape
    assert np.asarray(mean_molar_mass).shape == pressures.shape
    assert not np.allclose(h2o_profile, h2o_profile[0])
    assert np.allclose(h2o_constant, h2o_constant[0])


def test_calculate_gas_mass_fractions_accepts_tuple_filling_species():
    pressures = np.logspace(-6.0, 2.0, 5)
    temperatures = np.full_like(pressures, 1000.0)

    mass_fractions, _ = petitRADTRANS.chemistry.core.calculate_gas_mass_fractions(
        pressures=pressures,
        temperatures=temperatures,
        imposed_mass_fractions={},
        line_species=(),
        cloud_species=(),
        filling_species=('H2', 'He'),
        use_equilibrium_chemistry=False,
    )

    assert set(mass_fractions.keys()) == {'H2', 'He'}
    assert np.asarray(mass_fractions['H2']).shape == pressures.shape
    assert np.asarray(mass_fractions['He']).shape == pressures.shape
    assert np.asarray(mass_fractions['H2']).dtype == np.asarray(mass_fractions['He']).dtype
