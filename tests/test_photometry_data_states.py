from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from petitRADTRANS.retrieval.data import PhotometryData
from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig
from petitRADTRANS.retrieval.runtime import PhotometryObservationState


def test_photometry_data_materializes_observation_and_likelihood_state():
    def _synthetic_photometry(wavelengths, fluxes):
        del wavelengths
        return jnp.array([jnp.mean(fluxes)])

    data = PhotometryData(
        name="photometric_dataset",
        model_generating_function=lambda *args, **kwargs: None,
        photometric_transformation_function=_synthetic_photometry,
        photometric_bin_edges=(1.0, 2.0),
        radtrans_object=SimpleNamespace(name="radtrans"),
    )
    data.spectrum = jnp.array([0.1])
    data.uncertainties = jnp.array([0.01])
    data.mask = jnp.array([False])

    observation_state = data.to_observation_state(model_contract="differentiable")
    likelihood_state = data.to_likelihood_state(n_free_parameters=2)

    assert isinstance(data, PhotometryData)
    assert isinstance(observation_state, PhotometryObservationState)
    assert observation_state.model_contract == "differentiable"
    assert observation_state.photometric_transformation_function is _synthetic_photometry
    np.testing.assert_array_equal(
        observation_state.metadata["photometric_bin_edges"],
        jnp.array([1.0, 2.0]),
    )
    assert (
        observation_state.metadata["photometric_transformation_function_name"]
        == "_synthetic_photometry"
    )

    assert likelihood_state.n_free_parameters == 2
    assert likelihood_state.metadata["photometry"] is True
    np.testing.assert_array_equal(
        likelihood_state.metadata["photometric_bin_edges"],
        jnp.array([1.0, 2.0]),
    )


def test_add_data_infers_photometry_data_from_photometric_arguments():
    def _synthetic_photometry(wavelengths, fluxes):
        del wavelengths
        return jnp.array([jnp.mean(fluxes)])

    configuration = RetrievalConfig(retrieval_name="photometry_inference")
    configuration.add_data(
        name="synthetic_photometry",
        path_to_observations=None,
        model_generating_function=lambda *args, **kwargs: None,
        radtrans_object=object(),
        photometric_transformation_function=_synthetic_photometry,
        photometric_bin_edges=(1.0, 2.0),
        wavelengths=jnp.array([1.5]),
        spectrum=jnp.array([0.0]),
        uncertainties=jnp.array([0.1]),
    )

    assert isinstance(configuration.data["synthetic_photometry"], PhotometryData)