=====================================
petitRADTRANS 4: most notable changes
=====================================

Preface
=======
petitRADTRANS 4 is the first release series that replaces the legacy Fortran
radiative-transfer backend with a JAX implementation. Compared to
pRT3, this is a larger-than-usual upgrade: the numerical
backend, the retrieval interface, the optional sampler stack, and several public
parameter names all changed.

The JAX backend enables JIT compilation, automatic differentiation, and
execution on CPUs, GPUs, and TPUs through the JAX ecosystem. It also means that
pRT 4 is not numerically identical to version 3: small numerical
differences relative to pRT 3 are expected, and the 4.0.0a31 release should
still be treated as experimental.

Hereafter is a summary of the most important additions, behavioural changes,
and migration points between pRT 3.3.3 and pRT 4.0.0a31.

Added
=====
-  A JAX-based radiative-transfer backend. The translated core kernels now live
    in ``petitRADTRANS.radtrans_core``, and ``Radtrans``, chemistry, opacities,
    and math routines now operate on JAX arrays.
-  Native support for JIT compilation, automatic differentiation, and
    accelerator execution through JAX. This is the main architectural change in
    pRT 4 and is what enables GPU and TPU execution.
-  Autodiff-oriented spectrum APIs with
    ``Radtrans.calculate_flux_autodiff()`` and
    ``Radtrans.calculate_transit_radii_autodiff()``.
-  A new retrieval layer built around ``RetrievalRuntime``. Forward
    models can now follow a JAX-friendly contract instead of relying
    only on the legacy mutable ``Radtrans`` retrieval pattern.
-  Built-in differentiable JAX retrieval models exported from
    ``petitRADTRANS.retrieval.models``, together with dedicated tutorials for
      emission, transmission, and time-series retrievals.
-  New retrieval samplers: ``dynesty``, ``jaxns``, ``blackjax`` HMC and NUTS,
    and ``numpyro`` HMC and NUTS, in addition to the existing PyMultiNest and
    UltraNest workflows.
-  Ability to use optimal estimation to find the maximum a posteriori (MAP) solution
    using `optax`,which supports both gradient-based optimization and
    gradient-free methods.
-  A dedicated sampler layer that normalizes sampler selection, sample
    archiving, and summary generation across backends.
-  Covariance-aware retrievals, including Gaussian processes, automatic
    covariance hyperparameters in ``RetrievalConfig``, and Cholesky-based
    likelihood evaluation through ``LogLikelihood``.
-  The ``RetrievalPlotter`` class, which separates retrieval plotting from the
    core ``Retrieval`` object.
-  A new ``petitRADTRANS.experimental.sbi`` package scaffold for simulation-based inference,
    including simulators, training tasks, and normalising flow
    based SBI workflows.
-  Time-variable retrieval support through
    ``RetrievalConfig.add_time_series_data()`` and time-series
    model parameterizations.
-  Additional physics and modelling options, including non-vertically
    constant abundance profiles, isotope-ratio driven chemistry support for
    isotopologues, a JAX reimplementation of the exo-k correlated-k binning
    algorithm, Feautrier angle-dependent intensities, and JAX-compatible
    contribution-function evaluation.
-  Retrieval output summaries now include more sampler-native products, and the
    retrieval pipeline can compute model-selection diagnostics such as BIC, AIC,
    and BPIC.
-  New tutorial and reference material covering the JAX runtime, sampler
    choices, differentiable retrieval models, covariance-aware retrievals, and
    SBI.

Changed
=======
-  ``Radtrans`` now evaluates spectra through the JAX backend rather than the
    legacy Fortran implementation. In practice, this is a backend replacement,
    not just an optimization pass.
-  ``radtrans.py`` has been substantially refactored. Many static helpers were
    moved into dedicated modules, especially under
    ``petitRADTRANS.radtrans_core``, to separate physics kernels from the
    high-level API.
-  Spectrum calculations sanitize and normalize inputs before evaluation so
    they can be traced by JAX. This primarily affects users writing custom model
    wrappers or composing pRT inside larger JAX programs.
-  The retrieval workflow now supports both the legacy retrieval interface and
    the newer interface. The scalar and JAX execution paths now
    run through ``RetrievalRuntime`` when possible.
-  Memory management for line-by-line opacities and covariance handling has
    been reworked: pRT delays some JAX conversions, loads lbl data more
    selectively, and uses Cholesky factorization for large covariance matrices.
-  Transmission spectra and non-scattering emission calculations are now
    compatible with 32-bit execution modes, which matters for accelerator
    workflows where ``float32`` is often the default.
-  Feautrier emission calculations can now skip the scattering solve when no
    scattering opacity is present, and can optionally return angle-dependent
    intensities.
-  Partial cloud coverage is now expressed through the patchy-cloud interface.
    This is the main cloud-coverage surface documented in pRT 4.
-  The tutorial set, retrieval examples, and public documentation now assume
    the current v4 dataset-control names, the differentiable model examples, and
    the expanded sampler ecosystem.
-  pRT 4 ships optional dependency groups for ``[retrieval]``, ``[sbi]``, and
    ``[full]``, reflecting the broader runtime and inference stack introduced in
    this release line.
-  The installation and runtime expectations changed with the new backend: pRT
    4 depends on JAX, newer NumPy and Astropy versions, and a newer Python
    baseline.
-  Because the numerical core was rewritten, small numerical differences
    relative to the 3.x series are expected. The v4 documentation currently
    describes differences on the order of about 0.1 percent relative to earlier
    versions.

Breaking changes
================
-  **Python requirement:** the minimum supported Python version is now 3.12.
-  **Backend expectation:** custom code that relied on the old Fortran
    execution model, mutable Fortran-side state, or NumPy-only execution
    assumptions must be updated for a JAX-based backend.
-  **Retrieval dataset controls renamed:** ``scale`` -> ``scale_flux``,
    ``scale_err`` -> ``scale_uncertainties``, and ``offset_bool`` ->
    ``fit_flux_offset``.
-  **Emission geometry renamed:** ``emission_geometry`` is now
    ``irradiation_geometry`` throughout the main radiative-transfer API.
-  **Cloud particle argument names standardized:** canonical argument names now
    use ``cloud_particle_...`` instead of ``cloud_particles_...``.
-  **Patchy-cloud controls updated:** ``complete_coverage_clouds`` is no longer
    part of the canonical API. Use ``patchy_clouds`` to specify which clouds are
    patchy.
-  **Chemistry module rename:** ``condensation_curves.py`` is now
    ``vapour_pressures.py``.
-  **SpectralModel key rename:** the canonical key in
    ``SpectralModel.model_parameters`` is now ``modification_arguments`` instead
    of ``modification_parameters``.
-  **Retrieval chi-squared interface changed:** ``Data.get_chi2`` has been
    removed; chi-squared and likelihood evaluation now live in ``LogLikelihood``.
-  **Time-variable retrieval configuration moved:** legacy variability
    arguments were removed from ``RetrievalConfig.add_data()``; time-variable
    workflows must use ``add_time_series_data()``.
-  **Sampler requirements changed:** JAX-based samplers such as ``jaxns``,
    ``blackjax``, and ``numpyro`` require differentiable or otherwise
    JAX-traceable model groups. A legacy model that only works through the old
    mutable retrieval contract is not sufficient for every v4 sampler backend.
-  **Experimental status:** pRT 4.0.0a31 is still an alpha release. Existing
    pRT 3 production workflows should be revalidated after migration.

Removed
=======
-  ``Data.get_chi2`` has been removed in favour of ``LogLikelihood``.
-  Retrieval-time variability controls formerly passed through
    ``RetrievalConfig.add_data()`` have been removed from that interface and
    replaced by ``add_time_series_data()``.
-  The pRT2 -> pRT3 opacity-conversion workflow is not part of the pRT4
    migration story, because pRT4 does not introduce a new opacity file format.

Installation and runtime
------------------------

-  Upgrade to Python 3.12 or newer before migrating.
-  Expect newer dependency floors, including NumPy 2.0+, Astropy 7.0+, and
    JAX as a core dependency.
-  Install a JAX build that matches your hardware target. CPU-only
    installations work out of the box, while GPU or TPU execution depends on the
    JAX wheel you install.
-  Choose the optional extras that match your workflow:
    ``petitRADTRANS[retrieval]`` for retrievals,
    ``petitRADTRANS[sbi]`` for SBI, or ``petitRADTRANS[full]`` for the full
    stack.

Updating forward models
-----------------------

-  If your custom code only calls ``Radtrans.calculate_flux()`` or
    ``Radtrans.calculate_transit_radii()``, the high-level usage pattern remains
    familiar, but the returned arrays now come from a JAX-based backend.
-  If your forward model touches low-level internals, assumes NumPy-only
    arrays, or depends on the old Fortran kernels, update it to be JAX-safe or
    explicitly convert outputs with ``numpy.asarray(...)`` outside traced or JIT
    sections.
-  For differentiable retrievals and JAX-native samplers, prefer the
    differentiable model contract exposed through ``RetrievalRuntime`` and the
    built-in functions in ``petitRADTRANS.retrieval.models``.

Updating retrieval configurations
---------------------------------

-  Rename dataset-control flags to ``scale_flux``, ``scale_uncertainties``, and
    ``fit_flux_offset``.
-  Replace ``Data.get_chi2`` usage with ``LogLikelihood`` or the higher-level
    ``Retrieval`` helpers that now wrap it.
-  Use ``add_time_series_data()`` instead of passing variability-specific
    arguments through ``add_data()``.
-  If you plan to use ``jaxns``, ``blackjax``, or ``numpyro``, ensure that at
    least one model group is differentiable or otherwise fully JAX-traceable.

Updating physics and parameter names
------------------------------------

-  Rename ``emission_geometry`` to ``irradiation_geometry``.
-  Rename ``cloud_particles_*`` arguments and parameter keys to
    ``cloud_particle_*``.
-  Replace ``complete_coverage_clouds`` with ``patchy_clouds``.
-  Update imports from ``petitRADTRANS.chemistry.condensation_curves`` to
    ``petitRADTRANS.chemistry.vapour_pressures``.
-  In ``SpectralModel.model_parameters``, use ``modification_arguments`` rather
    than ``modification_parameters``.

