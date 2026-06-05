.. _retrieval_parallelisation:

=============================
Parallelising pRT Retrievals
=============================

This page explains how to run each retrieval sampler backend in parallel on HPC
hardware. It covers four hardware scenarios:

#. :ref:`Many CPU cores on a single node <par_cpu_single_node>`
#. :ref:`CPU cores spread across several nodes <par_cpu_multi_node>`
#. :ref:`One or more GPUs <par_gpu>`
#. :ref:`A CPU-driven sampler with a GPU-evaluated likelihood <par_cpu_gpu>`


For the list of samplers and their tuning parameters see
:ref:`retrieval_samplers`. The mechanism that offloads a single likelihood call
to an accelerator is documented at :ref:`gpu_likelihood_dispatch`; this page
explains *when* to use it.

.. note::

   The SLURM directives below target a generic Slurm cluster (the examples use
   the MPCDF "Vera" partitions). Adjust ``--partition``, ``--gres``, module
   names and memory to your site.


Two layers of parallelism
==========================

A pRT retrieval has two independent places where work can be parallelised, and
each sampler uses one or both:

**Sampler-level parallelism**
   Many likelihood evaluations are run concurrently. *How* depends on the
   backend family:

   * **Nested samplers** (PyMultiNest, UltraNest) evaluate the live-point
     proposals in parallel. PyMultiNest and UltraNest do this with **MPI ranks**,
     so they scale across cores *and* nodes.
   * **Dynesty** parallelises with a **Python multiprocessing pool**
     (``dynesty.pool.Pool``) — many worker *processes* on one node.
   * **JAX nested samplers** (JAXNS) and **gradient MCMC** (NumPyro, BlackJAX)
     parallelise with **JAX device sharding** (``shard_map`` / ``vmap`` over a
     device mesh) — naturally a multi-GPU mechanism.

**Likelihood-level parallelism**
   A *single* forward-model evaluation is spread across hardware. There are two
   sub-cases:

   * **Intra-op threading**: one JAX computation on one device uses many CPU
     threads (or the many cores of a GPU) internally. This is automatic.
   * **Device dispatch** (``likelihood_devices``): a CPU-resident sampler hands
     each scalar likelihood call to a JAX accelerator. Only PyMultiNest and
     UltraNest (non-vectorised) support this — see :ref:`par_cpu_gpu`.

Understanding which layer a sampler uses explains every compatibility statement
below.


Capability matrix
=================

.. list-table::
   :widths: 22 18 18 16 26
   :header-rows: 1

   * - Sampler
     - CPU / 1 node
     - CPU / multi-node
     - GPU(s)
     - CPU sampler + GPU likelihood
   * - ``pymultinest``
     - MPI ranks
     - **Yes** (MPI)
     - Kernel is CPU-only; offload likelihood
     - **Yes** (``likelihood_devices``)
   * - ``ultranest``
     - MPI ranks, or ``vectorized``
     - **Yes** (MPI)
     - ``vectorized`` on GPU, or offload likelihood
     - **Yes** (non-vectorised ``likelihood_devices``)
   * - ``dynesty``
     - Process pool (``pool_njobs``)
     - **No** (pool is single-node)
     - Not supported
     - Not supported
   * - ``jaxns``
     - Device mesh (limited on CPU)
     - Advanced only (multi-host)
     - **Yes** (``devices``)
     - N/A (whole sampler is on-device)
   * - ``jaxnsshardedstaticnestedsampler``
     - Device mesh (limited on CPU)
     - Advanced only (multi-host)
     - **Yes** (sharded)
     - N/A
   * - ``numpyronuts`` / ``numpyrohmc``
     - Chains via mesh (limited) or independent processes
     - Independent processes only
     - **Yes** (1 chain / GPU)
     - N/A (autodiff couples model to sampler)
   * - ``blackjaxnuts`` / ``blackjaxhmc``
     - Chains via mesh (limited) or independent processes
     - Independent processes only
     - **Yes** (1 chain / GPU)
     - N/A
   * - ``optimal_estimation`` / ``optax`` / ``map``
     - Single process (+intra-op threads)
     - **No**
     - **Yes** (runs on the JAX device)
     - N/A

"Limited on CPU" is explained in :ref:`par_xla_cpu_devices`: forcing several CPU
"devices" does **not** create real parallel hardware.


.. _par_building_blocks:

Building blocks: SLURM, MPI, XLA, threads
=========================================

These knobs recur in every scenario, so they are defined once here.

SLURM geometry
--------------

* ``--nodes=M`` — number of nodes.
* ``--ntasks-per-node=N`` — number of **MPI ranks** per node. Use this for
  PyMultiNest/UltraNest.
* ``--cpus-per-task=C`` — cores given to **one** task. Use this for a single
  process that parallelises internally (Dynesty pool, JAX intra-op threads, a
  single GPU job).
* ``--gres=gpu:<type>:K`` — request ``K`` GPUs.

The product ``ntasks-per-node × cpus-per-task`` must not exceed the cores on a
node. A pure-MPI sampler uses ``ntasks-per-node=N`` with ``cpus-per-task=1``; a
single threaded process uses ``ntasks-per-node=1`` with ``cpus-per-task=C``.

MPI
---

PyMultiNest and UltraNest use ``mpi4py`` and the MultiNest/UltraNest MPI paths.
Launch them so that each rank is a real MPI process with a valid PMI
environment:

.. code-block:: bash

   srun python my_retrieval.py        # SLURM provides PMI to every rank

Set ``use_MPI=True`` (the default) for PyMultiNest. Always pin BLAS/OpenMP to one
thread per rank so ranks do not oversubscribe cores:

.. code-block:: bash

   export OMP_NUM_THREADS=1

.. warning::

   The JAX samplers (JAXNS, NumPyro, BlackJAX) do **not** use MPI. If you launch
   them as plain processes that nevertheless import ``mpi4py`` through pRT,
   ``MPI_Init`` can abort with ``PMI2_Job_GetId returned 14`` when there is no
   per-process PMI environment. Block MPI for those launches **before importing
   petitRADTRANS**:

   .. code-block:: python

      import sys
      sys.modules["mpi4py"] = None   # forces pRT's no-MPI fallback (rank 0)

XLA
---

JAX-based samplers and the JAX likelihood are configured through JAX/XLA, not
MPI:

* **64-bit precision** (retrievals essentially always need this):

  .. code-block:: python

     import jax
     jax.config.update("jax_enable_x64", True)

* **Select the platform.** To make a GPU visible while keeping CPU available for
  the orchestration layer:

  .. code-block:: bash

     export JAX_PLATFORMS=cuda,cpu     # first entry is the default device

* **Per-process GPU isolation** (multi-GPU): give each process its own device

  .. code-block:: bash

     export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

* **Forcing CPU "devices"** (testing the sharded code paths only — see the
  warning in :ref:`par_xla_cpu_devices`):

  .. code-block:: bash

     export XLA_FLAGS="--xla_force_host_platform_device_count=8"

  This must be set **before** ``import jax``.

CPU threads for a single JAX device
-----------------------------------

A single JAX computation on CPU parallelises through XLA's intra-op thread pool,
which is independent of ``OMP_NUM_THREADS``. Under a Slurm cpuset the safest way
to bound it to the allocation is to confine the process (``taskset`` /
``--cpu-bind``) and let XLA's pool fill the visible cores. Do **not** set
``OMP_NUM_THREADS=1`` if you are relying on a single device using many cores for
BLAS-heavy work.


.. _par_xla_cpu_devices:

Why "many CPU devices" is not real CPU parallelism
==================================================

It is tempting to set ``--xla_force_host_platform_device_count=N`` and shard a
JAX sampler across ``N`` CPU devices. **This rarely speeds anything up.** Those
are *logical* devices that all time-share the same physical cores; sharding
across them adds collective/copy overhead without adding compute. The flag
exists to *exercise* multi-device code paths, not to gain performance.

On a CPU node, genuine parallelism comes from:

* **MPI ranks** (PyMultiNest, UltraNest),
* a **process pool** (Dynesty), or
* **independent OS processes**, one per MCMC chain (NumPyro, BlackJAX — see
  :ref:`par_independent_chains`),

and, within a single process, from **XLA intra-op threads** on *one* device.
Reserve device sharding for real accelerators.


.. _par_cpu_single_node:

Scenario 1 — Many CPU cores on a single node
============================================

PyMultiNest (recommended for nested sampling on CPU)
----------------------------------------------------

One MPI rank per core. SLURM:

.. code-block:: bash

   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=64
   #SBATCH --cpus-per-task=1

   export OMP_NUM_THREADS=1
   srun python my_retrieval.py

.. code-block:: python

   rc = RetrievalConfig(retrieval_name="run", run_mode="retrieval",
                        pressures=pressures, sampler_type="pymultinest")
   retrieval = Retrieval(rc, model_generating_function=my_model)
   retrieval.run(n_live_points=400, use_MPI=True, resume=False)

Scaling is near-linear in the number of live-point proposals up to a few hundred
ranks.

UltraNest
---------

Two options, which can be combined:

* **MPI ranks** — launch exactly like PyMultiNest (``srun`` with
  ``ntasks-per-node``); UltraNest detects ``mpi4py`` automatically.
* **Vectorised batches** — if the retrieval is fully differentiable, evaluate a
  whole batch of proposals in one JAX call:

  .. code-block:: python

     retrieval.run(min_num_live_points=400, vectorized=True)

  Vectorisation uses one process; combine it with a few MPI ranks rather than
  hundreds.

Dynesty (single node only)
--------------------------

Dynesty parallelises with a process pool. Request the cores as **one task** and
let pRT build the pool:

.. code-block:: bash

   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=32

.. code-block:: python

   retrieval.run(
       nlive=800, dlogz_init=0.05, use_jit=True,
       pool_njobs=32, queue_size=32,
       use_pool={"prior_transform": False, "loglikelihood": True,
                 "propose_point": True, "update_bound": True},
   )

``queue_size`` should match ``pool_njobs``; do not pass ``pool`` and
``pool_njobs`` together.

NumPyro / BlackJAX on CPU
-------------------------

A *single* NUTS/HMC chain is intrinsically sequential — extra cores can only help
through XLA intra-op threading of each gradient evaluation, which saturates well
before a full node. Two cores-using strategies exist:

* **Device-mesh chains** (``num_chains > 1`` with forced CPU devices): works, but
  see :ref:`par_xla_cpu_devices` — little real speedup on shared cores. Also note
  that **vectorising NUTS chains is inefficient**: a batched ``while_loop`` runs
  until the *longest* trajectory in the batch finishes, so every chain pays for
  the slowest one. HMC (fixed integration length) vectorises cleanly; NUTS does
  not.
* **Independent processes** (recommended): one OS process per chain, each pinned
  to a disjoint core set — see :ref:`par_independent_chains`. This avoids both the
  shared-core problem and the vectorised-NUTS penalty.

JAXNS on CPU
------------

JAXNS runs, but multi-CPU-device sharding is subject to the same shared-core
limitation. On a CPU node prefer a single device and let XLA thread it. Use JAXNS
on CPU mainly for correctness/development; use it on GPU for performance.


.. _par_cpu_multi_node:

Scenario 2 — CPU cores across several nodes
===========================================

This is the domain of the MPI nested samplers.

PyMultiNest / UltraNest (recommended)
-------------------------------------

Increase ``--nodes`` and keep one rank per core:

.. code-block:: bash

   #SBATCH --nodes=4
   #SBATCH --ntasks-per-node=64       # 256 ranks total
   #SBATCH --cpus-per-task=1

   export OMP_NUM_THREADS=1
   srun python my_retrieval.py

The sampler code is unchanged from the single-node case; MPI spans nodes
transparently under ``srun``.

**Incompatible: Dynesty.** pRT builds a ``dynesty.pool.Pool`` (Python
multiprocessing), which is confined to a single node. There is no built-in
multi-node Dynesty path. Use PyMultiNest or UltraNest across nodes instead.

NumPyro / BlackJAX / JAXNS across nodes
---------------------------------------

There is **no built-in multi-host (jax.distributed) path** for these backends in
pRT. The portable way to use several nodes is the
:ref:`independent-process pattern <par_independent_chains>`: launch independent
single-chain processes (across nodes via ``srun`` job steps or one submission per
node) and merge the posteriors afterwards. Each process is self-contained and
needs no inter-node communication, so it scales trivially across nodes — bounded
only by aggregate memory.


.. _par_gpu:

Scenario 3 — One or more GPUs
=============================

The JAX backends are the GPU-native samplers.

Single GPU
----------

.. code-block:: bash

   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=8
   #SBATCH --gres=gpu:a100:1

   export JAX_PLATFORMS=cuda,cpu

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

   # JAXNS:
   retrieval.run(num_live_points=400, devices=jax.devices("gpu"), use_jit=True)

   # NumPyro / BlackJAX, single chain on the GPU:
   retrieval.run(num_samples=2000, num_warmup=1000, num_chains=1)

Multiple GPUs
-------------

Request ``K`` GPUs and let JAX see all of them; the sampler shards across the
device mesh.

.. code-block:: bash

   #SBATCH --gres=gpu:a100:4
   #SBATCH --cpus-per-task=32

   export JAX_PLATFORMS=cuda,cpu

* **JAXNS** — pass the device list:

  .. code-block:: python

     retrieval.run(num_live_points=800, devices=jax.devices("gpu"), use_jit=True)

* **NumPyro / BlackJAX** — set ``num_chains`` equal to the number of GPUs so the
  ``shard_map`` path places **one chain per device** (``chains_per_device == 1``).
  This is the efficient mapping for NUTS because each device runs its own chain
  with its own trajectory length — no batched-``while_loop`` penalty:

  .. code-block:: python

     retrieval.run(num_samples=2000, num_warmup=1000,
                   num_chains=4)          # == number of GPUs

  .. note::

     On JAX builds that put the device-mesh axes in *Explicit* sharding mode, the
     multi-device NumPyro path passes the initial position into ``shard_map`` as
     an argument rather than closing over it; closing over an Explicit-sharded
     array raises ``NotImplementedError: Closing over inputs to shard_map ...``.
     This is handled inside pRT; no user action is required.

**Incompatible: Dynesty** (no GPU path) and the **MPI nested samplers' kernels**
(PyMultiNest/UltraNest run their search on the host — only their *likelihood* can
use a GPU, see :ref:`par_cpu_gpu`).


.. _par_cpu_gpu:

Scenario 4 — CPU sampler with a GPU likelihood
==============================================

Here the **sampler search runs on CPU** (often across many MPI ranks) while each
**forward-model evaluation is offloaded to a GPU**. This is the ``likelihood_devices``
mechanism, and it is supported **only** by PyMultiNest and UltraNest
(non-vectorised). See :ref:`gpu_likelihood_dispatch` for the implementation.

.. code-block:: bash

   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=8        # 8 CPU MPI ranks drive the sampler
   #SBATCH --cpus-per-task=1
   #SBATCH --gres=gpu:a100:2          # 2 GPUs evaluate likelihoods

   export OMP_NUM_THREADS=1
   export JAX_PLATFORMS=cuda,cpu
   srun python my_retrieval.py

.. code-block:: python

   import jax
   retrieval.run(
       n_live_points=400,
       use_MPI=True,
       likelihood_devices=jax.devices("gpu"),   # e.g. [CudaDevice(0), CudaDevice(1)]
   )

Each MPI rank *r* selects ``likelihood_devices[r % len(likelihood_devices)]``, so
the 8 ranks above stripe across the 2 GPUs (ranks 0,2,4,6 → GPU 0; ranks 1,3,5,7
→ GPU 1). For true isolation (no contention on a shared GPU) pin one device per
rank:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES=$(( SLURM_LOCALID % 2 ))

**Requirements and limits:**

* The retrieval must be **differentiable** (JAX forward model). pRT raises a
  ``ValueError`` at setup for legacy/non-JAX model groups.
* **UltraNest vectorised mode ignores** ``likelihood_devices`` (with a
  ``RuntimeWarning``): the vectorised path already dispatches through JAX and
  follows the standard JAX default-device selection. Choose *either*
  ``vectorized=True`` *or* ``likelihood_devices``.
* **Not applicable to the gradient MCMC and JAXNS backends.** Their likelihood is
  inside an autodiff/sharded computation that already runs on the chosen JAX
  device; there is no separate CPU "sampler" layer to keep on the host. To use a
  GPU with these, run the *whole* sampler on the GPU (:ref:`par_gpu`).
* **Not applicable to Dynesty** (no ``likelihood_devices`` support).


.. _par_independent_chains:

Pattern: independent-process MCMC chains
========================================

Because a single NUTS/HMC chain cannot use many cores, and device sharding gives
no speedup on CPU, the scalable way to run NumPyro/BlackJAX on CPU hardware — on
one node *or across nodes* — is **process-level parallelism**: launch ``N``
independent single-chain processes, each

* with a **distinct seed** (independent chains),
* with a **distinct output name** (no file collisions),
* **pinned to a disjoint core set** (``taskset`` / ``--cpu-bind``),
* with **``mpi4py`` blocked** (these are not MPI ranks; see the warning in
  :ref:`par_building_blocks`),

then **merge the posteriors** by concatenating the per-chain sample arrays.

This pattern respects a large per-evaluation memory footprint (one resident model
copy per process, never multiplied as ``vmap``-over-chains would) and needs no
inter-process communication, so the same launcher works across nodes. A minimal
launcher skeleton:

.. code-block:: bash

   N=4
   CORES=$(python -c 'import os;print(len(os.sched_getaffinity(0)))')
   PER=$(( CORES / N ))
   for c in $(seq 0 $((N-1))); do
       start=$(( c * PER )); end=$(( start + PER - 1 ))
       PRT_CHAIN_ID=$c PRT_WORKER=1 OMP_NUM_THREADS=$PER \
           taskset -c "${start}-${end}" python my_retrieval.py &
   done
   wait
   python combine_chains.py      # concatenate the per-chain posteriors

In the worker script, derive the per-chain seed and name from ``PRT_CHAIN_ID`` and
block MPI when ``PRT_WORKER=1``:

.. code-block:: python

   import os, sys
   if os.environ.get("PRT_WORKER") == "1":
       sys.modules["mpi4py"] = None          # before importing petitRADTRANS

   chain = int(os.environ.get("PRT_CHAIN_ID", "0"))
   seed = BASE_SEED + chain
   retrieval_name = f"{BASE_NAME}_chain{chain}"

To combine, load each chain's ``out_NumPyro/<name>_samples.npz`` and concatenate
along the sample axis; pRT's generic ``get_samples`` returns each chain as an
``(n_free + 1, n_samples)`` array (parameters followed by the log-density row).


Quick reference
===============

* **Nested sampling, scale-out (1 node or many):** PyMultiNest or UltraNest with
  MPI ranks (``srun``, ``ntasks-per-node``, ``OMP_NUM_THREADS=1``).
* **Nested sampling, one node, pure Python:** Dynesty with ``pool_njobs`` (single
  node only).
* **Nested sampling on GPU:** JAXNS with ``devices=jax.devices("gpu")``.
* **Gradient MCMC on GPU(s):** NumPyro/BlackJAX with ``num_chains == #GPUs``.
* **Gradient MCMC on CPU / across nodes:** independent-process chains + merge.
* **CPU search + GPU likelihood:** PyMultiNest/UltraNest with
  ``likelihood_devices=jax.devices("gpu")`` (non-vectorised).
* **Avoid:** ``--xla_force_host_platform_device_count`` for *performance*;
  vectorised NUTS across many chains; expecting Dynesty/JAX-MCMC to span nodes via
  the built-in paths.
