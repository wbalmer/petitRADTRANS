"""Lazy imports of the optional third-party libraries the samplers wrap.

Every sampler backend (JAX, JAXNS, BlackJAX, NumPyro, PyMultiNest, UltraNest,
Dynesty, ArviZ, matplotlib) is optional: a user installs only the ones they
intend to run. This module holds one ``require_<library>()`` function per
backend that imports it on first use and publishes its handles as **module
globals**, so importing any sampler module stays free.

.. important::

   Always reach these handles through the module object::

       from petitRADTRANS.retrieval.samplers import _optional_imports as _opt

       _opt.require_jax()
       array = _opt.jnp.zeros(3)

   Never ``from ._optional_imports import jnp``. That binds the name to
   ``None`` at import time and the rebinding done by ``require_jax()`` will
   never be seen, producing a ``'NoneType' object is not callable`` failure
   deep inside a run. In a long function body, rebind once locally after the
   require call instead::

       _opt.require_jax()
       jnp, random, jit = _opt.jnp, _opt.random, _opt.jit
"""

# --------------------------------------------------------------------------- #
# JAX
# --------------------------------------------------------------------------- #
lax = None
random = None
tree = None
vmap = None
tree_util = None
jit = None
local_device_count = None
make_mesh = None
shard_map = None
set_mesh = None
devices = None
default_device = None
device_get = None
clear_caches = None
jnp = None
P = None


def require_jax():
    """Import JAX and publish the handles the samplers use."""
    global lax, random, tree, vmap, tree_util, jit, local_device_count
    global make_mesh, shard_map, set_mesh, devices, default_device, device_get
    global clear_caches, jnp, P

    if jnp is not None:
        return

    try:
        from jax import (
            lax as _lax,
            random as _random,
            tree as _tree,
            vmap as _vmap,
            tree_util as _tree_util,
            jit as _jit,
            local_device_count as _ldc,
            make_mesh as _mm,
            shard_map as _sm,
            set_mesh as _set_mesh,
            devices as _devices,
            default_device as _default_device,
            device_get as _device_get,
            clear_caches as _clear_caches,
        )
        import jax.numpy as _jnp
        from jax.sharding import PartitionSpec as _ps
    except ImportError as error:
        raise ImportError(
            "JAX is required for JAX-based samplers but is not installed."
        ) from error

    lax = _lax
    random = _random
    tree = _tree
    vmap = _vmap
    tree_util = _tree_util
    jit = _jit
    local_device_count = _ldc
    make_mesh = _mm
    shard_map = _sm
    set_mesh = _set_mesh
    devices = _devices
    default_device = _default_device
    device_get = _device_get
    clear_caches = _clear_caches
    jnp = _jnp
    P = _ps


# --------------------------------------------------------------------------- #
# JAXNS 3 (+ the optional distributed runtime)
# --------------------------------------------------------------------------- #
jaxns = None
Model = None
NestedSampler = None
DistributedNestedSampler = None
NestedSamplerResults = None
DepthCondition = None
UniDimSliceSampler = None
# Private JAXNS helper, needed only when pRT synthesises a Python goal (see
# JAXNSSampler._build_goal_condition). It stays None when JAXNS moves it, and
# the one caller that needs it raises with migration guidance at that point,
# so an ordinary run never depends on a private symbol.
depth_condition_reached = None


def require_jaxns():
    """Import JAXNS 3 and the JAXCTX prior machinery it builds models from."""
    global jaxns, Model, NestedSampler, NestedSamplerResults
    global DepthCondition, UniDimSliceSampler, depth_condition_reached

    if jaxns is not None:
        return

    require_jax()

    # jaxns.mixed_precision flips jax_enable_x64 on at import time (and warns).
    # Importing it under a float32 pRT configuration would therefore silently
    # promote the whole forward model mid-run, so refuse up front instead.
    from petitRADTRANS.retrieval.parameter import FEAUTRIER_JAX_DTYPE

    if FEAUTRIER_JAX_DTYPE == jnp.float32:
        raise RuntimeError(
            "JAXNS 3 requires 64-bit JAX: importing it sets jax_enable_x64=True, "
            "which would silently change this float32 retrieval's forward model. "
            "Enable x64 before building the retrieval "
            "(jax.config.update('jax_enable_x64', True))."
        )

    # JAXNS 3 has kept changing its public surface *after* the 3.0.0 release
    # without bumping its version (its pyproject still reads 3.0.0), so a
    # version pin cannot express what pRT needs. Probe the actual API instead:
    # jaxns.depth_condition replaced jaxns.termination_condition, and is the
    # marker separating the 3.0.0 release from the API pRT targets.
    try:
        import jaxns as jaxns_module
        from jaxns.core import NestedSampler as jaxns_nested_sampler
        from jaxns.model import Model as jaxns_model
        from jaxns.results import NestedSamplerResults as jaxns_results
        from jaxns.depth_condition import DepthCondition as jaxns_depth_condition
        from jaxns.constrained_sampler import (
            UniDimSliceSampler as jaxns_unidim_slice_sampler,
        )
    except Exception as error:
        raise ImportError(
            "JAXNS 3 (and jaxctx) could not be imported. This sampler cannot be used. "
            "pRT requires a JAXNS 3 that exposes 'jaxns.depth_condition.DepthCondition'. "
            "A JAXNS 2.x install will fail here, and so will the JAXNS 3.0.0 release, "
            "which still had 'jaxns.termination_condition' -- note that JAXNS reports "
            "itself as 3.0.0 in both cases, so the version number cannot be trusted. "
            "Install jaxctx>=1.2.0 and a JAXNS 3 from after the depth-scheduling "
            "change (jaxns commit 4c70caf or later)."
        ) from error

    jaxns = jaxns_module
    Model = jaxns_model
    NestedSampler = jaxns_nested_sampler
    NestedSamplerResults = jaxns_results
    DepthCondition = jaxns_depth_condition
    UniDimSliceSampler = jaxns_unidim_slice_sampler

    # Best-effort: NestedSampler.run() uses this to turn the depth condition
    # into its default stopping goal. pRT needs the same predicate to keep
    # dlogZ terminating a run that also carries user goal criteria.
    try:
        from jaxns.algorithm.depth import _depth_condition_reached
    except Exception:
        depth_condition_reached = None
    else:
        depth_condition_reached = _depth_condition_reached


def require_jaxns_distributed():
    """Import the opt-in JAXNS distributed runtime (needs jaxns[distributed])."""
    global DistributedNestedSampler

    require_jaxns()

    if DistributedNestedSampler is not None:
        return

    try:
        from jaxns.distributed_core import (
            DistributedNestedSampler as jaxns_distributed_nested_sampler,
        )
        # The runtime client only needs these at registration time, but failing
        # here gives a clearer message than a mid-run ImportError.
        import cloudpickle  # noqa: F401
        import zmq  # noqa: F401
    except Exception as error:
        raise ImportError(
            "The JAXNS distributed backend requires the optional runtime "
            "dependencies. Install with `pip install 'jaxns[distributed]'` "
            "(cloudpickle and pyzmq)."
        ) from error

    DistributedNestedSampler = jaxns_distributed_nested_sampler


# --------------------------------------------------------------------------- #
# Gradient-based backends
# --------------------------------------------------------------------------- #
blackjax = None


def require_blackjax():
    global blackjax

    if blackjax is not None:
        return

    try:
        import blackjax as _blackjax
    except ImportError as error:
        raise ImportError(
            "BlackJAX is not installed. This sampler cannot be used."
        ) from error

    blackjax = _blackjax


numpyro = None
NumPyroHMC = None
NumPyroNUTS = None
NumPyroMCMC = None


def require_numpyro():
    global numpyro, NumPyroHMC, NumPyroNUTS, NumPyroMCMC

    if numpyro is not None:
        return

    try:
        import numpyro as _numpyro
        from numpyro.infer import HMC as _hmc
        from numpyro.infer import MCMC as _mcmc
        from numpyro.infer import NUTS as _nuts
    except ImportError as error:
        raise ImportError(
            "NumPyro is not installed. This sampler cannot be used."
        ) from error

    numpyro = _numpyro
    NumPyroHMC = _hmc
    NumPyroNUTS = _nuts
    NumPyroMCMC = _mcmc


# --------------------------------------------------------------------------- #
# Nested-sampling backends
# --------------------------------------------------------------------------- #
pymultinest = None


def require_pymultinest():
    global pymultinest

    if pymultinest is not None:
        return

    try:
        import pymultinest as _pymultinest
    except ImportError as error:
        raise ImportError(
            "PyMultiNest is not installed. This sampler cannot be used."
        ) from error

    pymultinest = _pymultinest


ultranest = None
MLFriends = None


def require_ultranest():
    global ultranest, MLFriends

    if ultranest is not None:
        return

    try:
        import ultranest as _ultranest
        from ultranest.mlfriends import MLFriends as _MLFriends
    except ImportError as error:
        raise ImportError(
            "UltraNest is not installed. This sampler cannot be used."
        ) from error

    ultranest = _ultranest
    MLFriends = _MLFriends


dynesty = None
DynamicNestedSampler = None
DynestyPool = None
dyplot = None
dyfunc = None


def require_dynesty():
    global dynesty, DynamicNestedSampler, DynestyPool, dyplot, dyfunc

    if dynesty is not None:
        return

    try:
        import dynesty as _dynesty
        from dynesty import DynamicNestedSampler as _DynamicNestedSampler
        from dynesty.pool import Pool as _DynestyPool
        from dynesty import plotting as _dyplot
        from dynesty import utils as _dyfunc
    except ImportError as error:
        raise ImportError(
            "Dynesty is not installed. This sampler cannot be used."
        ) from error

    dynesty = _dynesty
    DynamicNestedSampler = _DynamicNestedSampler
    DynestyPool = _DynestyPool
    dyplot = _dyplot
    dyfunc = _dyfunc


# --------------------------------------------------------------------------- #
# Diagnostics / plotting
# --------------------------------------------------------------------------- #
az = None
plt = None


def require_arviz():
    global az

    if az is not None:
        return

    try:
        import arviz as _az
    except ImportError as error:
        raise ImportError(
            "ArviZ is required to plot NumPyro diagnostics but is not installed."
        ) from error

    az = _az


def require_matplotlib_pyplot():
    global plt

    if plt is not None:
        return

    try:
        import matplotlib.pyplot as _plt
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to save sampler diagnostics plots but is not installed."
        ) from error

    plt = _plt
