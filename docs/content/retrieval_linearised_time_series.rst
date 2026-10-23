.. _retrieval_linearised_time_series:

==================================
Linearised Time-Series Retrievals
==================================

A rotating brown dwarf or directly imaged planet shows a spectrum that changes
with rotational phase. The natural way to retrieve that variability -- let the
atmospheric parameters vary sinusoidally in time and fit everything at once --
adds a large number of strongly correlated amplitude parameters to an already
difficult posterior, and samplers converge poorly as a result.

The variability is small, of order one percent in flux. To first order the
model is therefore *linear* in the modulation, and a linear model with Gaussian
priors and Gaussian errors has a posterior that can be written down in closed
form. The amplitudes need not be sampled at all: they are integrated out
analytically, and the sampler explores only the static reference state.

This page describes how to set such a retrieval up, how to recover the
modulation once it has run, and which diagnostics must accompany a published
result. For a worked example, from generating a synthetic SIMP 0136 time series
to recovering the injected modulation, see the
`"Linearised Time-Series Retrievals" tutorial <notebooks/retrieval_time_series.html>`_.

The statistical model
=====================

The fit is performed on the logarithm of the observed flux:

.. math::

   \ln d_{ij} = \ln M_i(\theta_\mathrm{ref}) + a_i
              + \sum_{pm} J_{ip}\,\Phi_{jm}\,\beta_{pm} + \varepsilon_{ij}

for wavelength :math:`i` and epoch :math:`j`, where

- :math:`M_i(\theta_\mathrm{ref})` is the reference model spectrum, projected
  onto the observation grid;
- :math:`a_i \sim \mathcal{N}(0, s^2)` is a static per-wavelength offset
  carrying residual forward-model error. It is a nuisance parameter: never
  sampled, always marginalised;
- :math:`J_{ip} = \partial \ln M_i / \partial \theta_p` is the log Jacobian,
  the fractional change in flux per unit change in parameter :math:`p`. Each
  column is a *response spectrum*;
- :math:`\Phi_{jm}` is the harmonic basis in time, anchored on a reference set
  of epochs so that :math:`\theta_\mathrm{ref}` is the state at a defined
  rotational phase;
- :math:`\beta_{pm}` are the modulation amplitudes, the quantity of interest;
- :math:`\varepsilon_{ij}` is measurement noise, independent in time by
  default, or AR(1)-correlated if a correlation time is supplied.

Both :math:`a_i` and :math:`\beta_{pm}` are Gaussian and enter linearly, so both
are integrated out exactly. What the sampler evaluates is
:math:`\ln Z(\theta_\mathrm{ref})`, a function of the static state alone.

Marginalising the offset leaves an effective noise covariance
:math:`\Sigma_i = S_i + s^2\,\mathbf{1}\mathbf{1}^\mathsf{T}` at each
wavelength, which is diagonal-plus-rank-one and is inverted in closed form by
the Sherman-Morrison identity. No :math:`(n_\mathrm{time}, n_\mathrm{time})`
matrix is ever formed.

Setting up a retrieval
======================

The model function must return the reference spectrum *and* its flux Jacobian.
Wrap any runtime-native model with
:func:`~petitRADTRANS.retrieval.models.make_linearised_time_series_model`:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)  # required, see below

   from petitRADTRANS.retrieval.models import (
       gradient_profile_emission,
       make_linearised_time_series_model,
   )
   from petitRADTRANS.retrieval.retrieval_config import RetrievalConfig

   model = make_linearised_time_series_model(
       gradient_profile_emission,
       jacobian_mode="jacfwd",
   )

   configuration = RetrievalConfig(
       retrieval_name="my_variability_run",
       run_mode="retrieval",
       pressures=pressures,
       sampler_type="jaxns",
   )
   # ... add the static parameters and the rotation period ...

   configuration.add_linearised_time_series_data(
       name="NIRSPEC",
       observation_times=times,
       model_generating_function=model,
       modulated_parameters={"T_bottom": 500.0, "log_X_H2O": 2.0},
       n_harmonics=3,
       static_offset_scale=0.05,
       wavelengths=wavelengths,
       spectrum=spectrum,
       uncertainties=uncertainties,
   )

``modulated_parameters`` is a mapping rather than a list because each prior
width enters the marginal likelihood directly and is therefore part of the
model definition. The interface does not allow it to be supplied by omission.

Key arguments
-------------

``n_harmonics``
    Highest retained harmonic of the rotation frequency. Because the visible
    hemisphere is averaged over, the signal is band-limited and only the first
    few harmonics carry information.

``static_offset_scale``
    The fractional static forward-model error tolerated, :math:`s`. A free
    per-wavelength offset with this prior width absorbs any static spectral
    error, however it is correlated in wavelength, and simultaneously supplies
    the absolute-flux constraint. ``None`` gives an infinitely wide offset:
    shape-only inference, with the absolute flux discarded.

``noise_correlation_time``
    Correlation time of the noise in units of epochs, modelled as an AR(1)
    process. ``None`` is independent noise. The AR(1) inverse is tridiagonal in
    closed form, so correlated noise costs no more than uncorrelated noise.

``nuisance_response_modes``
    Number of smooth wavelength modes appended to the log Jacobian as extra
    response spectra. The per-wavelength offset absorbs static model error
    exactly; these absorb error that varies in *time* and is smooth in
    wavelength, which a frozen-cloud model cannot express. Mode 0 is grey.
    Their amplitudes are marginalised alongside the physical ones, so the model
    remains exactly linear-Gaussian and the physical amplitudes simply acquire
    the extra uncertainty that comes from not knowing how much of the signal a
    smooth component could have produced.


Recovering the modulation
=========================

Because the amplitudes are marginalised, the sampler output contains only the
static parameters. At any given reference state the amplitude posterior is an
exact Gaussian, so it is reconstructed rather than sampled:

.. code-block:: python

   posterior = retrieval.calculate_modulation_posterior(parameters="best fit")

   posterior["amplitudes"]                    # (n_response, 2 * n_harmonics)
   posterior["covariance"]                    # (n_column, n_column)
   posterior["column_names"]                  # e.g. ('T_bottom:c1', 'T_bottom:s1', ...)
   posterior["basis"]                         # (n_time, 2 * n_harmonics)
   posterior["excursion"]                     # (n_time, n_response)
   posterior["excursion_standard_deviation"]  # the 1-sigma band

The result is *conditional* on the reference state passed in. The amplitudes
depend on that state far more strongly than on the measurement noise, so a
single call understates the uncertainty. For the unconditional posterior,
evaluate this at a set of draws from the reference-state posterior and combine
them with :func:`~petitRADTRANS.retrieval.linearised.mixture_moments`, which
splits the total covariance by the law of total variance:

.. math::

   \mathrm{Cov}(\beta) = \underbrace{\langle C_s \rangle_s}_{\text{statistical}}
                       + \underbrace{\mathrm{Cov}_s(\hat\beta_s)}_{\text{reference state}}

Each draw contributes an exact Gaussian rather than a single point, which makes
this a Rao-Blackwellised estimate and correspondingly less noisy than sampling
the amplitudes would have been. The conditional posteriors must be computed at
*every* draw: fixing the reference state and solving once afterwards discards
the second term entirely.

:meth:`~petitRADTRANS.retrieval.retrieval.Retrieval.plot_modulation` does all of
this and draws the result: it solves the amplitude posterior at
``n_reference_samples`` draws from the posterior, combines them, and plots the
excursion of each modulated parameter with the two bands kept separate, so that
the reference-state term remains visible rather than being folded into a single
error envelope. It is called automatically by
:meth:`~petitRADTRANS.retrieval.retrieval.Retrieval.plot_all` for every
registered linearised dataset. Each reference state costs one forward model and
one Jacobian, which is why the solved amplitudes are cached next to the figure
and reused unless ``refresh=True``.

.. code-block:: python

   sample_dict, parameter_dict = retrieval.get_samples()
   parameters_read = parameter_dict[retrieval.configuration.retrieval_name]

   retrieval.plot_modulation(sample_dict, parameters_read, n_reference_samples=50)

Four more plots draw on the same results:

* :meth:`~petitRADTRANS.retrieval.retrieval.Retrieval.plot_modulation_correlation`
  shows the correlations of the amplitudes, in total and split into the
  statistical and the reference-state parts. It reads the amplitudes cached by
  ``plot_modulation``.
* :meth:`~petitRADTRANS.retrieval.retrieval.Retrieval.plot_modulation_temperature`
  shows how the temperature profile departs from its rotation mean, against
  rotational phase and pressure. It shifts the modulated parameters by their
  excursion and evaluates the model's own temperature profile, so it works for
  any temperature parameterisation, and draws nothing if no modulated parameter
  changes the profile.
* :meth:`~petitRADTRANS.retrieval.retrieval.Retrieval.plot_modulation_light_curves`
  and :meth:`~petitRADTRANS.retrieval.retrieval.Retrieval.plot_modulation_map`
  compare the data with the fitted modulation at one reference state, the best
  fit by default: as light curves in a few wavelength bands, and as the
  spectrum divided by its time average, folded on the rotation, for the data,
  the model and their difference. Their shared solve is cached too, so pass
  ``refresh=True`` after changing the reference state.

All four are called by
:meth:`~petitRADTRANS.retrieval.retrieval.Retrieval.plot_all` as well.

.. code-block:: python

   retrieval.plot_modulation_correlation(sample_dict, parameters_read)
   retrieval.plot_modulation_temperature(sample_dict, parameters_read)
   retrieval.plot_modulation_light_curves()
   retrieval.plot_modulation_map()

Diagnostics
===========

Three checks should accompany any published result.

**The linearisation gate.** What biases an amplitude is not the size of the
linearisation error but its projection onto the design.
:func:`~petitRADTRANS.retrieval.linearised.amplitude_bias_projection` evaluates
that projection: pass the difference between the exact and linearised models in
log space, and every returned value should be small compared with one. The same
function answers a second question if given a fit residual instead: how far an
unmodelled systematic could move each amplitude.

**Prior-width sensitivity.** A marginal likelihood is a statement about a model
*including* its priors. Widening all amplitude priors by a factor :math:`f`
moves :math:`\ln Z` by :math:`M \ln f` for :math:`M` amplitudes, which can
exceed any physical difference being sought. Run
:func:`~petitRADTRANS.retrieval.linearised.evidence_prior_scan` for every
variant being compared and report a conclusion only if the *ordering* survives
across the range of widths that are physically defensible. Where it does not,
:func:`~petitRADTRANS.retrieval.linearised.predictive_log_density` answers the
same question without depending on the prior width at all.

**Conditioning.** An individual amplitude is meaningful only if the data can
tell it apart from the others. Where two response spectra look alike, the data
constrain a combination tightly and each amplitude hardly at all, and
per-amplitude error bars then overstate what was measured.
:func:`~petitRADTRANS.retrieval.linearised.posterior_eigenspectrum` reports the
directions the data actually determine.

Further reading
===============

The module :mod:`petitRADTRANS.retrieval.linearised` documents the algebra in
detail, including the Sherman-Morrison and Woodbury identities used, the
treatment of an infinitely wide offset as a restricted likelihood, and the
conventions for array shapes and amplitude ordering.
