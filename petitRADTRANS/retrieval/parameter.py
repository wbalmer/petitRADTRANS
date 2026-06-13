import jax.numpy as jnp
from jax import config, tree_util
from petitRADTRANS.jax_dtype import JAX_DTYPE  # noqa: F401  (re-exported for other modules)
from petitRADTRANS.retrieval.utils import (
    log_prior,
    uniform_prior,
    gaussian_prior,
    log_gaussian_prior,
    delta_prior,
)


def _cast_distribution_to_jax_dtype(distribution):
    """
    Re-create a TFP distribution with floating-point parameters cast to the active JAX dtype.
    """
    target_dtype = jnp.float64 if config.jax_enable_x64 else jnp.float32
    if hasattr(distribution, "dtype") and distribution.dtype == target_dtype:
        return distribution

    params = {}
    for param_name, param_value in distribution.parameters.items():
        if param_value is None:
            continue
        try:
            arr = jnp.asarray(param_value)
            if jnp.issubdtype(arr.dtype, jnp.floating):
                params[param_name] = jnp.asarray(param_value, dtype=target_dtype)
            else:
                params[param_name] = param_value
        except (TypeError, ValueError):
            params[param_name] = param_value

    try:
        return distribution.__class__(**params)
    except Exception:
        return distribution


def _build_jaxns_prior(distribution, name):
    try:
        from jaxns import Prior
    except Exception as error:
        raise ImportError(
            "JAXNS is required to build Prior objects for JAXNS retrievals, but it could not be imported."
        ) from error

    return Prior(distribution, name=name)


class Parameter:
    """
    Parameter
    This class provides the interface for defining and managing parameters in a retrieval.
    Each parameter includes a name, which can be used as a reference in the model function,
    a value, and flag of whether it's a free parameter, and if it's free. You can also define
    a function that translates the unit hypercube into physical space for a pymultinest retrieval,
    or a tensorflow probability distribution to define the prior for JAX-based retrievals.
    The remainder of the arguments deal with the corner plots.

    Args:
        name : string
            The name of the parameter. Must match the name used in the model function.
        is_free_parameter : bool
            True if the parameter should be sampled in the retrieval
        value : float
            The value of the parameter. Set using set_param.
        transform_prior_cube_coordinate : method
            Transform the unit interval [0,1] to the physical space of the parameter.
        plot_in_corner : bool
            True if this parameter should be included in the output corner plot
        corner_ranges : Tuple(float,float)
            The axis range of the parameter in the corner plot
        corner_transform : method
            A function to scale or transform the value of the parameter for prettier plotting.
        corner_label : string
            The axis label for the parameter, defaults to name.
        units : astropy.units.UnitBase, optional
            The physical units of the parameter value. Defaults to None, in which case the
            value is assumed to already be in the units expected by the model function (cgs).
            If set to an astropy unit, the model functions can use it to convert the value to
            the units required for the radiative transfer calculation (e.g. cm, g).
    """

    def __init__(
        self,
        name,
        is_free_parameter,
        value=0.0,
        distribution=None,
        transform_prior_cube_coordinate=None,
        plot_in_corner=False,
        corner_ranges=None,
        corner_transform=None,
        corner_label=None,
        units=None,
    ):

        self.name = name
        self.is_free_parameter = is_free_parameter
        self.value = value
        self.distribution = (
            _cast_distribution_to_jax_dtype(distribution)
            if distribution is not None
            else None
        )
        self.prior_sample = None
        self.transform_prior_cube_coordinate = transform_prior_cube_coordinate
        self.plot_in_corner = plot_in_corner
        self.corner_ranges = corner_ranges
        self.corner_transform = corner_transform
        self.corner_label = corner_label
        self.units = units

    def _tree_flatten(self):
        children = (self.value,)  # arrays / dynamic values
        aux_data = {
            "name": self.name,
            "is_free_parameter": self.is_free_parameter,
            "distribution": self.distribution,
            "transform_prior_cube_coordinate": self.transform_prior_cube_coordinate,
            "plot_in_corner": self.plot_in_corner,
            "corner_ranges": self.corner_ranges,
            "corner_transform": self.corner_transform,
            "corner_label": self.corner_label,
            "units": self.units,
        }  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(value=children[0], **aux_data)

    def get_param_uniform(self, cube):
        if self.is_free_parameter:
            if self.transform_prior_cube_coordinate is not None:
                return self.transform_prior_cube_coordinate(cube)

            if self.distribution is not None:
                return self.distribution.quantile(cube)

            raise ValueError(
                f"Error! Free parameter '{self.name}' has neither a prior transform "
                f"nor a distribution with a quantile method"
            )

        raise ValueError(f"Error! Parameter '{self.name}' is not a free parameter")

    def get_flattened_value(self, value=None):
        if value is None:
            value = self.value

        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            _value = []

            keys = list(value.keys())
            values = list(value.values())

            for i, v in enumerate(values):
                v = self.get_flattened_value(value=v)
                k = self.get_flattened_value(value=[keys[i]])
                flattened_dict = jnp.concatenate((k, v))
                _value.append(flattened_dict.flatten())
        else:
            _value = value

        return jnp.array(_value).flatten()

    def set_param(self, value):
        if self.is_free_parameter:
            self.value = value
            return

        raise ValueError(f"Error! Parameter '{self.name}' is not a free parameter")

    def set_prior(self):
        return _build_jaxns_prior(self.distribution, name=self.name)
        # return self.prior_sample


class RetrievalParameter:
    __available_priors = [
        "log",
        "uniform",
        "gaussian",
        "log_gaussian",
        "delta",
        "custom",
    ]

    def __init__(self, name, prior_parameters, prior_type="uniform", custom_prior=None):
        """Used to set up retrievals.
        Stores the prior function. Prior parameters depends on the type of prior. e.g., for uniform and log prior, these
        are the bounds of the prior. For gaussian priors and alike, these are the values of the mean and full width
        half maximum.

        Args:
            name: name of the parameter to retrieve, must match the corresponding model parameter of a SpectralModel
            prior_parameters: list of two values for the prior parameters, depends on the prior type
            prior_type: type of prior to use, the available types are stored into available_priors
            custom_prior: function with arguments (cube, *args), args being positional arguments in prior_parameters
        """
        # Check prior parameters validity
        if not hasattr(prior_parameters, "__iter__"):
            raise ValueError(
                f"'prior_parameters' must be an iterable of size 2, but is of type '{type(prior_parameters)}'"
            )
        elif isinstance(prior_parameters, jnp.ndarray) and jnp.size(prior_parameters) < 2:
            raise ValueError(
                f"'prior_parameters' must be of size 2, but is of size '{jnp.size(prior_parameters)}'"
            )
        elif len(prior_parameters) < 2:
            raise ValueError(
                f"'prior_parameters' must be of size 2, but is of size '{len(prior_parameters)}'"
            )
        elif prior_parameters[0] > prior_parameters[1] and (
            prior_type == "log" or prior_type == "uniform"
        ):
            raise ValueError(
                f"lower prior boundaries ({prior_parameters[0]}) "
                f"must be lower than upper prior boundaries ({prior_parameters[1]})"
            )

        self.name = name
        self.prior_parameters = prior_parameters
        self.prior_type = prior_type

        # Set prior
        if self.prior_type == "log":

            def prior(x):
                return log_prior(
                    cube=x, lx1=self.prior_parameters[0], lx2=self.prior_parameters[1]
                )

        elif self.prior_type == "uniform":

            def prior(x):
                return uniform_prior(
                    cube=x, x1=self.prior_parameters[0], x2=self.prior_parameters[1]
                )

        elif self.prior_type == "gaussian":

            def prior(x):
                return gaussian_prior(
                    cube=x, mu=self.prior_parameters[0], sigma=self.prior_parameters[1]
                )

        elif self.prior_type == "log_gaussian":

            def prior(x):
                return log_gaussian_prior(
                    cube=x, mu=self.prior_parameters[0], sigma=self.prior_parameters[1]
                )

        elif self.prior_type == "delta":

            def prior(x):
                return delta_prior(
                    cube=x,  # actually useless
                    x1=self.prior_parameters[0],
                    x2=self.prior_parameters[1],  # actually useless
                )

        elif self.prior_type == "custom":

            def prior(x):
                return custom_prior(cube=x, *prior_parameters)

        else:
            raise ValueError(
                f"prior type '{prior_type}' not implemented "
                f"(available prior types: {'|'.join(RetrievalParameter.__available_priors)})"
            )

        self.prior_function = prior

    @classmethod
    def from_dict(cls, dictionary):
        """Convert a dictionary into a list of RetrievalParameter.
        The keys of the dictionary are the names of the RetrievalParameter. The values of the dictionary must be
        dictionaries with keys 'prior_parameters' and 'prior_type'.

        Args:
            dictionary: a dictionary

        Returns:
            A list of RetrievalParameter.
        """
        new_retrieval_parameters = []

        for key, parameters in dictionary.items():
            new_retrieval_parameters.append(
                cls(
                    name=key,
                    prior_parameters=parameters["prior_parameters"],
                    prior_type=parameters["prior_type"],
                )
            )

        return new_retrieval_parameters

    def put_into_dict(self, dictionary=None):
        """Convert a RetrievalParameter into a dictionary.

        Args:
            dictionary: a dictionary; if None, a new dictionary is created

        Returns:
            A dictionary.
        """
        if dictionary is None:
            dictionary = {}

        dictionary[self.name] = {
            "prior_boundaries": self.prior_parameters,
            "prior_type": self.prior_type,
        }

        return dictionary


tree_util.register_pytree_node(
    Parameter, Parameter._tree_flatten, Parameter._tree_unflatten
)
