"""Sampler backends for petitRADTRANS retrievals, and the registry that finds them.

``RetrievalConfig.sampler_type`` is a string; :func:`resolve_sampler_class`
turns it into the :class:`~petitRADTRANS.retrieval.samplers.base.Sampler`
subclass that implements it.

To build a custom sampler, put the class in :data:`SAMPLER_DISPATCH` under
its ``sampler_type`` string; it takes precedence over the built-in table. See
the "Defining a custom sampler" section of the retrieval samplers
documentation.
"""

import importlib

from petitRADTRANS.retrieval.samplers.base import Sampler, SamplerContext

__all__ = [
    "SAMPLER_DISPATCH",
    "Sampler",
    "SamplerContext",
    "resolve_sampler_class",
]

#: ``sampler_type`` alias -> (module basename inside this package, class name).
#: The class is imported only when that alias is actually requested.
_SAMPLER_REGISTRY: dict[str, tuple[str, str]] = {
    "pymultinest": ("pymultinest", "PymultinestSampler"),
    "ultranest": ("ultranest", "UltranestSampler"),
    "dynesty": ("dynesty", "DynestySampler"),
    "blackjaxhmc": ("blackjax", "BlackJAXHMCSampler"),
    "blackjax_hmc": ("blackjax", "BlackJAXHMCSampler"),
    "hmc": ("blackjax", "BlackJAXHMCSampler"),
    "blackjaxnuts": ("blackjax", "BlackJAXNUTSSampler"),
    "blackjax_nuts": ("blackjax", "BlackJAXNUTSSampler"),
    "nuts": ("blackjax", "BlackJAXNUTSSampler"),
    "numpyrohmc": ("numpyro", "NumPyroHMCSampler"),
    "numpyro_hmc": ("numpyro", "NumPyroHMCSampler"),
    "numpyronuts": ("numpyro", "NumPyroNUTSSampler"),
    "numpyro_nuts": ("numpyro", "NumPyroNUTSSampler"),
    "jaxnestedsampler": ("jaxns", "JAXNSSampler"),
    "jaxns": ("jaxns", "JAXNSSampler"),
    "jaxnsdistributed": ("jaxns_distributed", "JAXNSDistributedSampler"),
    "jaxns_distributed": ("jaxns_distributed", "JAXNSDistributedSampler"),
    "optimal_estimation": ("optimal_estimation", "OptimalEstimationSampler"),
    "optimalestimation": ("optimal_estimation", "OptimalEstimationSampler"),
    "optax": ("optimal_estimation", "OptimalEstimationSampler"),
    "map": ("optimal_estimation", "OptimalEstimationSampler"),
}

#: Mutable registry for custom backends registered at runtime. Also used as the
#: resolution cache for the built-in table.
SAMPLER_DISPATCH: dict[str, type[Sampler]] = {}

def resolve_sampler_class(
    sampler_type: str,
    strict: bool = True,
) -> type[Sampler] | None:
    """Return the ``Sampler`` subclass implementing ``sampler_type``.

    Args:
        sampler_type: an alias from :data:`_SAMPLER_REGISTRY`, a key registered
            in :data:`SAMPLER_DISPATCH`, or a retired JAXNS 2 name.
        strict: whether a backend that cannot be imported is an error. Callers
            that *run* a retrieval want the ImportError, because the name they
            asked for is the one they need. Callers that merely *ask a backend
            about itself* -- which output folder it owns, whether it needs
            distribution priors -- must not fail merely because an unrelated
            optional dependency is absent, and pass ``strict=False`` to get
            ``None`` instead.
    Returns:
        The sampler class, or ``None`` if the name is not known (or, when
        ``strict`` is False, if its backend cannot be imported).
    Raises:
        ImportError: if the backend's own third-party library is missing and
            ``strict`` is True.
    """
    sampler_type_normalized = sampler_type.lower()

    sampler_class = SAMPLER_DISPATCH.get(sampler_type_normalized)
    if sampler_class is not None:
        return sampler_class

    registry_entry = _SAMPLER_REGISTRY.get(sampler_type_normalized)
    if registry_entry is None:
        return None

    module_basename, sampler_class_name = registry_entry
    try:
        sampler_module = importlib.import_module(
            f"petitRADTRANS.retrieval.samplers.{module_basename}"
        )
    except ImportError:
        if strict:
            raise
        return None
    sampler_class = getattr(sampler_module, sampler_class_name)
    SAMPLER_DISPATCH.setdefault(sampler_type_normalized, sampler_class)
    return sampler_class
