.. _retrieval_samplers:

==================
Retrieval Samplers
==================

This page summarizes the samplers available in the pRT retrieval package,
the most important keyword arguments for each, and the interface you need
to implement if you want to add your own sampler.

Sampler choice in pRT is controlled through ``RetrievalConfig.sampler_type``.
Sampler-specific keyword arguments are passed to :meth:`petitRADTRANS.retrieval.retrieval.Retrieval.run`.

For example, to run a Dynesty retrieval you would typically write:

.. code-block:: python

   rc = RetrievalConfig(
       retrieval_name="my_run",
       run_mode="retrieval",
       pressures=pressures,
       sampler_type="dynesty",
   )
   ...
   retrieval = Retrieval(rc)
   retrieval.run(nlive=800, dlogz_init=0.05, use_jit=True)

All built-in samplers follow the same high-level flow:

1. ``Retrieval`` builds a :class:`petitRADTRANS.retrieval.sampler.SamplerContext`.
2. The sampler class turns that context into a backend-specific likelihood and prior.
3. The backend runs and returns posterior samples.
4. ``Retrieval`` saves a generic ``out_samples/<retrieval_name>_samples.npz`` archive for downstream plotting and analysis.

If a backend is inherently weighted, it should convert its posterior to equal-weight samples before handing the results back to ``Retrieval``.
The built-in Dynesty and JAXNS integrations already do this.

Choosing a sampler
------------------

.. list-table::
   :widths: 22 18 22 38
   :header-rows: 1

   * - ``sampler_type``
     - Family
     - Requirements
     - Typical use
   * - ``pymultinest``
     - Nested sampling
     - MultiNest / PyMultiNest
     - Established MPI nested sampler for production nested-sampling workflows.
   * - ``ultranest``
     - Nested sampling
     - UltraNest
     - Pure-Python reactive nested sampling, especially convenient when you want warm starts or step sampling.
   * - ``dynesty``
     - Dynamic nested sampling
     - Dynesty
     - Dynamic nested sampling with pure-Python control flow and optional JIT-compiled likelihood evaluation.
   * - ``jaxns``
     - Nested sampling
     - JAXNS, JAX, TFP
     - Default JAX-native nested sampler for differentiable retrievals.
   * - ``jaxnsshardedstaticnestedsampler``
     - Nested sampling
     - JAXNS, JAX, TFP
     - Advanced static/sharded JAXNS backend for more specialized distributed runs.
   * - ``numpyronuts`` / ``numpyro_nuts``
     - Gradient MCMC
     - NumPyro, JAX
     - Adaptive NUTS for differentiable posteriors when you want efficient local exploration instead of evidence estimation.
   * - ``numpyrohmc`` / ``numpyro_hmc``
     - Gradient MCMC
     - NumPyro, JAX
     - Adaptive HMC when you want tighter control over integration length than NUTS provides.
   * - ``blackjaxnuts`` / ``blackjax_nuts``
     - Gradient MCMC
     - BlackJAX, JAX
     - JAX-native NUTS with explicit warmup control and a lightweight integration.
   * - ``blackjaxhmc`` / ``blackjax_hmc``
     - Gradient MCMC
     - BlackJAX, JAX
     - JAX-native HMC for differentiable models with explicit integration-step control.
   * - ``optimal_estimation`` / ``optax`` / ``map``
     - Optimisation (MAP + Laplace)
     - optax, JAX
     - Fast maximum-a-posteriori fit with an optional Laplace (Gaussian) posterior. Also the engine behind the ``optimize_init`` warm-start for the NUTS/HMC samplers.


Backend-specific parameters
---------------------------

The lists below focus on the parameters you are most likely to tune in practice.
Each backend may accept additional low-level keyword arguments that are forwarded
to the underlying sampler implementation.

PyMultiNest
~~~~~~~~~~~

Key parameters:

* ``n_live_points``: number of live points.
* ``sampling_efficiency``: trade-off between posterior estimation and evidence accuracy.
* ``const_efficiency_mode``: constant-efficiency mode, usually only for specialized runs.
* ``evidence_tolerance``: convergence tolerance on the evidence.
* ``resume``: continue an existing run instead of overwriting it.
* ``max_iter``: stop after a fixed number of iterations.
* ``multimodal``: enable mode finding.
* ``importance_nested_sampling``: enable INS mode.
* ``use_MPI``: use MPI if available.
* ``likelihood_devices``: list of JAX devices to dispatch likelihood evaluation to (see :ref:`gpu_likelihood_dispatch`).

Use PyMultiNest when you already rely on the MultiNest ecosystem or need its MPI-oriented execution model.

UltraNest
~~~~~~~~~

Key parameters:

* ``vectorized``: evaluate batches of samples at once when the retrieval is fully differentiable.
* ``resume``: reuse an existing UltraNest log directory.
* ``dlogz``: evidence convergence target.
* ``frac_remain``: stop when the remaining live-point weight drops below this fraction.
* ``min_num_live_points``: minimum number of live points.
* ``max_iters`` / ``max_ncalls``: hard stopping limits.
* ``enable_step_sampling``: attach an UltraNest step sampler.
* ``warmstart_max_tau``: enable warm starts from similar previous runs.
* ``likelihood_devices``: list of JAX devices to dispatch likelihood evaluation to (non-vectorized path only; see :ref:`gpu_likelihood_dispatch`).

UltraNest is often the easiest pure-Python nested sampler to experiment with when you want reactive nested sampling or warm-start support.

Dynesty
~~~~~~~

Key parameters:

* ``use_jit``: JIT-compile the scalar likelihood when the retrieval runtime is differentiable.
* ``nlive``: number of live points used by ``DynamicNestedSampler``.
* ``bound``: bounding strategy.
* ``sample``: proposal strategy.
* ``walks`` / ``slices`` / ``facc``: proposal-tuning parameters.
* ``periodic`` / ``reflective``: periodic or reflective dimensions.
* ``pool``: pass an existing Dynesty-compatible worker pool object.
* ``pool_njobs``: convenience option that asks pRT to build an internal ``dynesty.pool.Pool`` with this many worker processes.
* ``queue_size``: number of queued proposals. When pRT creates the pool internally, this defaults to the pool size if you do not pass it explicitly.
* ``use_pool``: Dynesty per-stage pool switches such as ``loglikelihood`` and ``propose_point``. If a pool is active and this is omitted, pRT defaults to using the pool for likelihood evaluation, point proposals, and bound updates, but not for the prior transform.
* ``dlogz_init``: initial evidence stopping threshold.
* ``nlive_init`` / ``nlive_batch``: live-point settings for the initial and follow-up batches.
* ``n_effective``: stop once an effective posterior sample target is reached.
* ``resume`` / ``checkpoint_file`` / ``checkpoint_every``: checkpoint and resume controls.

Dynesty is useful when you want dynamic nested sampling but prefer a Python-native backend while still reusing the pRT likelihood and prior wiring.
For parallel CPU runs, pRT can now construct ``dynesty.pool.Pool`` internally via ``pool_njobs`` and will wrap the retrieval likelihood/prior functions in pool-safe proxy callables before handing them to Dynesty.

For example, a parallel Dynesty run can be configured as:

.. code-block:: python

   retrieval.run(
     nlive=800,
     dlogz_init=0.05,
     use_jit=True,
     pool_njobs=8,
     queue_size=8,
     use_pool={
       "prior_transform": False,
       "loglikelihood": True,
       "propose_point": True,
       "update_bound": True,
     },
   )

In practice, ``queue_size`` should usually match the number of worker processes, and ``pool`` and ``pool_njobs`` should not be passed together.

JAXNS
~~~~~

Key parameters:

* ``num_live_points``: number of live points.
* ``max_samples``: hard cap on the number of nested-sampling evaluations.
* ``s``, ``k``, ``c``: JAXNS nested-sampling control parameters.
* ``shell_fraction``: shell-fragment control for proposal generation.
* ``gradient_guided``: enable gradient-guided proposals.
* ``parameter_estimation``: toggle estimation-oriented behavior.
* ``init_efficiency_threshold``: early-efficiency threshold.
* ``devices``: JAX devices to use.
* ``seed``: random seed.
* ``use_jit``: JIT the nested-sampler call.
* ``termination_condition``, ``term_cond``, or ``termination_condition_kwargs``:
  pass a mapping or JAXNS ``TerminationCondition``-like object with stopping
  criteria.
* Direct termination-condition fields are also accepted as keyword arguments:
  ``ess``, ``evidence_uncert``, ``live_evidence_frac``, ``dlogZ``,
  ``max_num_likelihood_evaluations``, ``log_L_contour``,
  ``efficiency_threshold``, ``rtol``, ``atol``, and ``peak_XL_frac``.
* ``termination_<field>`` aliases are accepted for all termination-condition
  fields. In practice, ``termination_max_samples`` is the important one because
  plain ``max_samples`` configures the nested sampler itself.

JAXNS is the default JAX-native nested sampler in pRT. It expects a differentiable retrieval runtime and now saves equal-weight posterior samples for downstream plotting.

If you do not supply any termination-condition settings, pRT leaves
``term_cond=None`` and JAXNS falls back to its internal defaults.

For example, this run keeps the sampler-level ``max_samples`` cap at ``50000``
while also building a termination condition with ``dlogZ=0.1``, ``ess=512``,
and a termination-condition ``max_samples`` of ``20000``:

.. code-block:: python

   retrieval.run(
       num_live_points=400,
       max_samples=50000,
       dlogZ=0.1,
       ess=512,
       termination_max_samples=20000,
       use_jit=True,
   )

You can also pass the same stopping criteria as a mapping:

.. code-block:: python

   retrieval.run(
       num_live_points=400,
       termination_condition_kwargs={
           "dlogZ": 0.1,
           "rtol": 1e-4,
           "peak_XL_frac": 1e-2,
       },
       use_jit=True,
   )

JAXNS sharded static sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key parameters:

* ``num_live_points``: number of live points.
* ``max_samples``: hard cap on the total number of samples.
* ``devices``: JAX devices to use.
* ``difficult_model``: hint for harder likelihood surfaces.
* ``verbose``: backend verbosity.
* ``seed``: random seed.
* ``use_jit``: JIT the sampler call.

This backend accepts the same termination-condition setup as ``jaxns``:
``termination_condition`` / ``term_cond`` / ``termination_condition_kwargs``,
direct termination-condition field kwargs, and ``termination_<field>`` aliases.
Prefer ``jaxns`` unless you specifically need the static sharded execution strategy.

NumPyro NUTS / HMC
~~~~~~~~~~~~~~~~~~

Key parameters shared by both NumPyro samplers:

* ``num_samples``: number of post-warmup draws.
* ``num_warmup``: warmup length.
* ``initial_step_size``: initial integrator step size.
* ``target_acceptance_rate``: acceptance target used during adaptation.
* ``is_mass_matrix_diagonal``: diagonal or dense mass matrix.
* ``num_chains``: number of chains.
* ``thinning``: chain thinning factor.
* ``chain_method``: NumPyro chain execution mode.
* ``seed``: random seed.
* ``use_jit``: accepted for API consistency; NumPyro still runs through its JIT-based execution path.

Additional HMC controls:

* ``num_integration_steps`` or ``num_steps``: leapfrog steps.
* ``trajectory_length``: alternative HMC trajectory specification.

NumPyro is a good choice when you want adaptive Hamiltonian MCMC rather than nested sampling and your retrieval is fully differentiable.

BlackJAX NUTS / HMC
~~~~~~~~~~~~~~~~~~~

Key parameters shared by both BlackJAX samplers:

* ``num_samples``: number of post-warmup draws.
* ``num_warmup``: warmup length.
* ``initial_step_size``: initial integrator step size.
* ``target_acceptance_rate``: adaptation target.
* ``is_mass_matrix_diagonal``: diagonal or dense mass matrix.
* ``num_chains``: number of chains.
* ``seed``: random seed.
* ``use_jit``: JIT the per-chain sampling loop.

Additional HMC control:

* ``num_integration_steps``: leapfrog steps for HMC.

BlackJAX is useful when you want a thinner wrapper around JAX-native HMC or NUTS with explicit warmup and multi-chain control.

.. _gpu_likelihood_dispatch:

GPU-dispatched likelihood for CPU samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyMultiNest and UltraNest (non-vectorized) are CPU-bound samplers: the sampler
kernel itself runs on the host and calls the likelihood function as a plain Python
callable.  When a GPU or other JAX-supported accelerator is available, you can
offload each likelihood evaluation to that device by passing a list of JAX devices
as ``likelihood_devices``.

.. code-block:: python

   import jax

   retrieval.run(
       n_live_points=400,
       likelihood_devices=jax.devices("gpu"),   # e.g. [CudaDevice(id=0)]
   )

pRT wraps the likelihood in a JIT-compiled call to ``evaluate_likelihood_jax``, dispatches the
computation to the target device using JAX's ``default_device`` context manager, and
returns a plain Python float to the sampler via ``jax.device_get``.  The sampler and
MPI orchestration layer stay on CPU throughout.

**Requirements:**

* The retrieval must use a differentiable (JAX-compatible) model.  pRT raises a
  ``ValueError`` at setup time if any model group uses a legacy (non-JAX) forward model.
* JAX must be able to see the target device.  For GPU runs you typically need
  ``JAX_PLATFORMS=cuda,cpu`` (or ``gpu,cpu``) in the environment so that both backends
  are initialised.

**MPI and multi-GPU mapping:**

When running with MPI (e.g. via ``mpirun``), each rank *r* independently selects
``likelihood_devices[r % len(likelihood_devices)]``.  With two GPUs and four MPI
ranks the default striping is:

.. code-block:: text

   rank 0 → GPU 0
   rank 1 → GPU 1
   rank 2 → GPU 0
   rank 3 → GPU 1

For true device isolation (no shared-memory contention between MPI processes on the
same GPU), set ``CUDA_VISIBLE_DEVICES`` per process before launching MPI, or use a
job launcher that handles device assignment automatically.

**UltraNest vectorized mode:**

When ``vectorized=True`` is also passed, ``likelihood_devices`` is ignored with a
``RuntimeWarning``: the vectorized path already dispatches through JAX
(``evaluate_likelihood_jax_batched``) and device selection there follows the standard JAX
default-device mechanism.

Optimal estimation (optax)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``optimal_estimation`` backend (aliases ``optax`` and ``map``) does not sample
the posterior with a Markov chain or nested sampling. Instead it uses `optax
<https://optax.readthedocs.io>`_ to find the maximum-a-posteriori (MAP) point by
gradient descent on the (negative) log-posterior, and — by default — builds a
Laplace approximation of the posterior from the curvature at that optimum. This is
the classic *optimal estimation* idea (Rodgers): near the mode the posterior is
approximately Gaussian with covariance equal to the inverse Hessian of the negative
log-posterior. It is implemented in
:mod:`petitRADTRANS.retrieval.optimal_estimation`.

Like the gradient-MCMC backends, it requires a fully differentiable retrieval
runtime and operates in the *unconstrained* parameter space (so all prior bounds are
respected automatically through the change-of-variables that pRT already applies).

Key parameters:

* ``optimizer``: the optax optimiser. Accepts ``None`` (defaults to Adam), an optax
  name string such as ``"adam"``, ``"adamw"``, or ``"adabelief"``, an already-built
  ``optax.GradientTransformation``, or a callable factory ``optimizer(learning_rate)``.
* ``learning_rate``: scalar or optax learning-rate schedule.
* ``clip_norm``: global-norm gradient-clipping threshold (prepended to the optimiser).
  This matters in practice: at the prior median a high-resolution petitRADTRANS
  log-posterior can have gradients of order :math:`10^{11}`, and clipping keeps the
  first steps bounded. Set to ``None`` to disable.
* ``n_steps``: maximum optimisation steps (alias ``num_steps``).
* ``tol``: optional gradient infinity-norm tolerance for early stopping.
* ``patience``: stop after this many steps without improvement (``None`` disables).
* ``n_starts`` / ``start_jitter``: number of random restarts and the jitter width for
  the extra starts (start 0 is always the prior median). The best optimum is kept,
  which guards against local optima.
* ``lbfgs_polish_steps``: if ``> 0``, refine the best optax optimum with this many
  ``optax.lbfgs`` (zoom line-search) iterations. This is recommended for tightening
  very flat or ill-conditioned directions that first-order methods converge slowly.
* ``compute_laplace``: if ``True`` (default), compute the inverse-Hessian covariance
  and draw a Gaussian posterior from it. If ``False``, the result is the single MAP
  point.
* ``n_posterior_samples``: number of draws from the Laplace Gaussian (the first draw
  is the MAP itself).
* ``seed``: random seed for restarts and the Laplace draws.

Outputs and caveats:

* Results are written to ``out_OptimalEstimation/<retrieval_name>_*`` (samples,
  parameters, covariance, and a metadata JSON with the MAP best fit), and the
  generic ``out_samples`` archive is saved as usual so the standard plotting and
  ``Retrieval.get_samples()`` paths work.
* For a Laplace run, ``samples`` are physical-space draws from the Gaussian posterior
  and ``log_likelihood`` is the *Gaussian* (Laplace) log-density of each draw — this
  avoids one forward-model evaluation per draw. The true model log-posterior at the
  MAP is reported separately as ``best_logdensity``.
* This is a *local* optimiser and a *Gaussian* posterior approximation: it is fast and
  ideal for a quick best fit, a sanity check, or an initial guess, but it will not
  capture multimodal or strongly non-Gaussian posteriors. Use a nested sampler or
  gradient-MCMC backend for the full posterior.
* If the Hessian at the optimum is not positive definite (a poor or saddle-like
  optimum), its eigenvalues are floored before inversion and a warning is emitted;
  treat the error bars as approximate in that case.

A typical run with multi-start, an L-BFGS polish, and a Laplace posterior:

.. code-block:: python

   rc = RetrievalConfig(
       retrieval_name="my_run",
       run_mode="retrieval",
       pressures=pressures,
       sampler_type="optimal_estimation",
   )
   ...
   retrieval = Retrieval(rc)
   retrieval.run(
       optimizer="adam",
       learning_rate=1e-2,
       clip_norm=1.0,
       n_steps=500,
       n_starts=4,
       lbfgs_polish_steps=50,
       compute_laplace=True,
       n_posterior_samples=2000,
   )

Warm-starting NUTS and HMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient-MCMC samplers are *local*: they explore from wherever they are initialised.
By default pRT initialises the NumPyro and BlackJAX walkers at the prior median
(the origin of the unconstrained space). For a poorly-constrained starting model this
can sit in a region where the log-posterior is extremely steep, so the very first
leapfrog step diverges and the chain never moves — a failure mode that shows up as a
near-zero acceptance rate, 100% divergences, and "samples" frozen at the prior median.

To avoid this, the NumPyro and BlackJAX backends accept an optional optax warm-up that
replaces the prior-median initial position with a MAP estimate before sampling starts.
Enable it with ``optimize_init=True`` and configure the optimiser through
``optimizer_kwargs`` (forwarded to
:func:`petitRADTRANS.retrieval.optimal_estimation.optimize_unconstrained_position`,
so it accepts the same ``optimizer``, ``learning_rate``, ``clip_norm``, ``n_steps``,
``patience``, ``n_starts``, and ``lbfgs_polish_steps`` keys as above). The optimiser
defaults to Adam.

.. code-block:: python

   rc = RetrievalConfig(
       retrieval_name="my_run",
       run_mode="retrieval",
       pressures=pressures,
       sampler_type="numpyro_nuts",   # or "blackjaxnuts", "numpyro_hmc", "blackjaxhmc"
   )
   ...
   retrieval = Retrieval(rc)
   retrieval.run(
       num_warmup=1000,
       num_samples=1500,
       target_acceptance_rate=0.9,
       optimize_init=True,
       optimizer_kwargs={
           "optimizer": "adam",
           "n_steps": 300,
           "learning_rate": 1e-2,
           "clip_norm": 1.0,
           "lbfgs_polish_steps": 50,
       },
   )

Notes:

* ``optimize_init`` is ignored if you pass your own ``initial_position`` explicitly.
* For multi-chain runs, add ``"walker_jitter": <sigma>`` to ``optimizer_kwargs`` to
  spread the chains around the MAP (chain 0 stays exactly at the optimum, the rest get
  Gaussian jitter of width ``sigma``). This keeps multi-chain convergence diagnostics
  such as R-hat meaningful.
* The warm-up seed defaults to the sampler ``seed``; override it with
  ``"seed": <int>`` inside ``optimizer_kwargs``.
* The achieved MAP log-posterior is recorded in the run summary as
  ``warmup_best_logdensity`` so you can confirm the warm-up moved the walkers into the
  bulk of the posterior.

Defining a custom sampler
-------------------------

Custom samplers should inherit from :class:`petitRADTRANS.retrieval.sampler.Sampler`.
For full integration with ``Retrieval.run()``, implement a ``prepare`` class method that consumes a
:class:`petitRADTRANS.retrieval.sampler.SamplerContext` and returns a sampler instance,
the keyword arguments for ``run_sampler()``, and a small dictionary of summary parameters.

The minimal contract is:

* ``prepare(context, **sampler_kwargs)`` returns ``(sampler, run_kwargs, summary_parameters)``.
* ``run_sampler(**run_kwargs)`` executes the backend and returns results.
* ``get_results()`` returns the in-memory results object.
* Optionally implement ``post_run()``, ``plot_diagnostics()``, ``load_results()``, and ``get_summary()``.

To participate cleanly in pRT posterior plotting and ``Retrieval.get_samples()``, your results object should expose:

* ``samples``: an equal-weight posterior matrix of shape ``(n_samples, n_parameters)``.
* ``log_likelihood``: one log-likelihood value per posterior sample.
* ``log_weights``: optional log weights. For equal-weight posteriors this is just ``-log(n_samples)``.

If your backend is naturally weighted, keep the weighted representation in an auxiliary ``weighted_samples`` block and return equal-weight samples to ``Retrieval`` for plotting.

Here is a minimal example:

.. code-block:: python

   import numpy as np

   from petitRADTRANS.retrieval.retrieval import SAMPLER_DISPATCH
   from petitRADTRANS.retrieval.sampler import Sampler, SamplerContext


   class MySampler(Sampler):
       def __init__(self, log_likelihood_func, prior_func, output_directory, retrieval_name):
           super().__init__(log_likelihood_func, prior_func, output_directory, retrieval_name)
           self.results = None

       @classmethod
       def prepare(cls, context: SamplerContext, **sampler_kwargs):
           sampler = cls(
               log_likelihood_func=context.log_likelihood_func,
               prior_func=context.prior_ultranest_func,
               output_directory=context.output_directory,
               retrieval_name=context.retrieval_name,
           )
           run_kwargs = {
               "n_draws": sampler_kwargs.pop("n_draws", 512),
               "seed": sampler_kwargs.pop("seed", 42654),
               "parameter_names": list(context.free_parameter_names),
           }
           run_kwargs.update(sampler_kwargs)
           summary_parameters = {
               "sampler_type": "mysampler",
               "n_draws": run_kwargs["n_draws"],
               "seed": run_kwargs["seed"],
               "parameter_names": run_kwargs["parameter_names"],
           }
           return sampler, run_kwargs, summary_parameters

       def run_sampler(self, n_draws=512, seed=42654, parameter_names=None, **kwargs):
           rng = np.random.default_rng(seed)
           cube_samples = rng.random((n_draws, len(parameter_names)))
           physical_samples = np.asarray([self.prior_func(sample) for sample in cube_samples])
           log_likelihood = np.asarray(
               [self.log_likelihood_func(sample) for sample in physical_samples],
               dtype=float,
           )

           self.results = {
               "samples": physical_samples,
               "log_likelihood": log_likelihood,
               "log_weights": np.full(n_draws, -np.log(n_draws), dtype=float),
           }
           return self.results

       def get_results(self):
           return self.results


   SAMPLER_DISPATCH["mysampler"] = MySampler

Two integration patterns are available:

* Register the class in ``SAMPLER_DISPATCH`` and set ``RetrievalConfig.sampler_type="mysampler"``.
  This is the preferred route because ``Retrieval.run()`` will call ``prepare()``, ``post_run()``, and the standard summary machinery.
* Pass an already-built instance with ``retrieval.run(sampler_object=my_sampler, ...)``.
  This is useful for one-off experiments, but it bypasses ``prepare()`` and ``post_run()``. In that mode you are responsible for any extra setup, summary information, and diagnostics you need.

Practical guidance
------------------

* Use ``jaxns``, NumPyro, or BlackJAX only with differentiable retrieval models.
* Use equal-weight posterior samples for downstream plotting, archive export, and best-fit selection.
* Keep backend-native outputs in a sampler-specific folder such as ``out_JAXNS`` or ``out_Dynesty`` if you need to reload diagnostics later.
* If you add a new sampler backend to pRT itself, update both ``SAMPLER_DISPATCH`` and the user-facing sampler documentation on this page.