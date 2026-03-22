==================
Retrieval Examples
==================

In this section we will give fully implemented examples of retrievals.

If you use the pRT retrieval package, please cite `Nasedkin et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024JOSS....9.5875N/abstract>`_.

The retrieval tutorials use the current dataset-control names `scale_flux`, `scale_uncertainties`, and `fit_flux_offset`.
They also include the runtime-native JAX retrieval path and covariance-aware workflows through the same `Data` and `RetrievalConfig` interfaces.

.. toctree::
    :maxdepth: 2

    notebooks/retrieval_basic
    notebooks/retrieval_jax_runtime_emission_tutorial
    notebooks/retrieval_multiple_data
    notebooks/retrieval_emission
    notebooks/retrieval_models
    notebooks/retrieval_spectral_model_jwst
    notebooks/retrieval_spectral_model
    notebooks/retrieval_advanced
