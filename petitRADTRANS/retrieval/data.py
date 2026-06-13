import os
import warnings
from typing import Callable

import numpy as np
import jax.numpy as jnp
from numpy import genfromtxt
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike
from astropy.io import fits

import petitRADTRANS.physical_constants as cst
from petitRADTRANS.retrieval.covariance import (
    normalize_covariance_mode,
    normalize_global_kernel_name,
    normalize_local_kernel_name,
)
from petitRADTRANS.retrieval.runtime import (
    LikelihoodState,
    ObservationState,
    PhotometryObservationState,
    _apply_flux_scaling,
    _offset_flux,
    _line_b_uncertainty_scaling,
    _scale_and_inflate_uncertainties,
)

from petitRADTRANS.jax_dtype import JAX_DTYPE


class Data:
    r"""
    This class stores the spectral data to be retrieved from a single instrument or observation.

    Each dataset is associated with an instance of petitRadTrans and an atmospheric model.
    The pRT instance can be overwritten, and associated with an existing pRT instance with the
    external_pRT_reference parameter.
    This setup allows for joint or independent retrievals on multiple datasets.
    Args:
        name : str
            Identifier for this data set.
        model_generating_function : method
            A function, typically defined in run_definition.py that returns the model wavelength and spectrum
            (emission or transmission).
            This is the function that contains the physics of the model, and calls pRT in order to compute the
            spectrum.
        path_to_observations : str
            Path to observations file, including filename. This can be a txt or dat file
            containing the wavelength, flux, transit depth and error, or a fits file
            containing the wavelength, spectrum and covariance matrix.
            Alternatively, the data information can be directly given by the wavelengths, spectrum, uncertainties, and
            mask attributes.
        wavelengths:
            (um) Wavelengths of the data.
        spectrum:
            Spectrum of the data.
        uncertainties:
            Uncertainties of the data, in the same units as the spectrum.
        covariance:
            Covariance matrix of the data, in the same units as the spectrum squared.
        line_opacity_mode : str
            Should the retrieval be run using correlated-k opacities (default, 'c-k'),
            or line by line ('lbl') opacities? If 'lbl' is selected, it is HIGHLY
            recommended to set the model_resolution parameter. In general,
            'c-k' mode is recommended for retrievals of everything other than
            high-resolution (R>40000) spectra.
        data_resolution : float or jnp.ndarray
            Spectral resolution of the instrument. Optional, allows convolution of model to
            instrumental line width. If the data_resolution is an array, the resolution can
            vary as as a function of wavelength. The array should have the same shape as
            the input wavelength array, and should specify the spectral resolution at each
            wavelength bin.
        model_resolution : float
            Will be ``None`` by default.  The resolution of the c-k opacity tables in pRT.
            This will generate a new c-k table using exo-k. The default (and maximum)
            correlated k resolution in pRT is :math:`\\lambda/\\Delta \\lambda > 1000` (R=500).
            Lowering the resolution will speed up the computation.
            If integer positive value, and if ``opacities == 'lbl'`` is ``True``, then this
            will sample the high-resolution opacities at the specified resolution.
            This may be desired in the case where medium-resolution spectra are
            required with a :math:`\\lambda/\\Delta \\lambda > 1000`, but much smaller than
            :math:`10^6`, which is the resolution of the ``lbl`` mode. In this case it
            may make sense to carry out the calculations with line_by_line_opacity_sampling = 10e5,
            for example, and then re-binning to the final desired resolution:
            this may save time! The user should verify whether this leads to
            solutions which are identical to the re-binned results of the fiducial
            :math:`10^6` resolution. If not, this parameter must not be used.
            Note the difference between this parameter and the line_by_line_opacity_sampling
            parameter in the RadTrans class - the actual desired resolution should
            be set here.
        scale_flux : bool
            Turn on or off scaling the data by a constant factor. Set to True if scaling the data during the
            retrieval.
        scale_uncertainties:
            Turn on or off scaling the uncertainties by a constant factor. Set to True if
            scaling the uncertainties during the retrieval.
        fit_flux_offset:
            Turn on or off fitting a flux offset. Set to True if fitting a flux offset during the retrieval.
        fit_covariance:
            If True, construct a fitted covariance matrix during retrieval using GP-like kernel terms.
            This is currently supported for 1D spectroscopic datasets only.
        covariance_mode:
            Covariance model to use. Supported values are ``'none'``, ``'local'``, ``'global'``, and
            ``'global_local'``.
        global_covariance_kernel:
            Global kernel name used when ``covariance_mode`` includes a global term.
        local_covariance_kernel:
            Local kernel name used when ``covariance_mode`` includes local terms.
        n_local_covariance_kernels:
            Number of local covariance kernels to include when using ``'global_local'`` mode.
        covariance_jitter:
            Small diagonal stabilization term added to the fitted covariance matrix.
        external_radtrans_reference : object
            An existing RadTrans object. Leave as none unless you're sure of what you're doing.
        wavelength_boundaries : tuple,list
            Set the wavelength range of the pRT object. Defaults to a range +/-5% greater than that of the data.
            Must at least be equal to the range of the data.
        wavelength_bin_widths : numpy.ndarray
            Set the wavelength bin width to bin the Radtrans object to the data. Defaults to the data bins.
        radtrans_grid: bool
            Set to true if data has been binned to a pRT c-k grid.
        radtrans_object:
            An instance of Radtrans object to be used to generate model spectra in retrievals.
        mask:
            Mask of the data.
    """

    resolving_power_str = ".R"

    def __init__(
        self,
        name: str,
        model_generating_function: (
            Callable[..., tuple[ArrayLike, ArrayLike, ArrayLike]] | None
        ) = None,
        path_to_observations: str | None = None,
        wavelengths: ArrayLike | None = None,
        spectrum: ArrayLike | None = None,
        uncertainties: ArrayLike | None = None,
        covariance: ArrayLike | None = None,
        line_opacity_mode: str = "c-k",
        data_resolution: float | None = None,
        model_resolution: float | None = None,
        scale_flux: bool = False,
        scale_uncertainties: bool = False,
        fit_flux_offset: bool = False,
        fit_covariance: bool = False,
        covariance_mode: str = "none",
        global_covariance_kernel: str = "squared_exponential",
        local_covariance_kernel: str = "squared_exponential",
        n_local_covariance_kernels: int = 0,
        covariance_jitter: float = 0.0,
        fit_instrumental_resolution: bool = False,
        external_radtrans_reference: object | None = None,
        wavelength_boundaries: tuple[float, float] | None = None,
        wavelength_bin_widths: ArrayLike | None = None,
        subtract_continuum: bool = False,
        radtrans_grid: bool = False,
        radtrans_object: object = None,
        mask: ArrayLike | None = None,  # TODO bool typing
    ):
        self.name: str = name
        self.path_to_observations: str | None = path_to_observations

        # To be filled later
        self.radtrans_object: object = radtrans_object
        self.wavelengths: ArrayLike | None = wavelengths  #: The wavelength bin centers
        self.spectrum: ArrayLike | None = spectrum  #: The flux or transit depth
        self.uncertainties: ArrayLike | None = (
            uncertainties  #: The error on the flux or transit depth
        )

        # Add a mask with that will be used in retrievals
        self.mask: ArrayLike

        if mask is None:
            self.mask = jnp.zeros(jnp.shape(self.spectrum), dtype=bool)
        else:
            self.mask = mask

        self.data_resolution: float | None = data_resolution
        self.data_resolution_array_model: ArrayLike | None = None
        self.variability_atmospheric_column_model_flux_return_mode: bool = False

        self.model_resolution: float | None = model_resolution
        self.external_radtrans_reference: object = external_radtrans_reference
        self.model_generating_function: Callable = model_generating_function
        self.line_opacity_mode: str = line_opacity_mode

        if line_opacity_mode not in ["c-k", "lbl"]:
            raise ValueError(
                f"line_opacity_mode must be either 'c-k' or 'lbl', but is '{line_opacity_mode}'"
            )

        # Sanity check model function
        if model_generating_function is None and external_radtrans_reference is None:
            raise ValueError(
                "any of parameters 'model_generating_function' or 'external_radtrans_reference' must be set, "
                "but both were None"
            )

        if model_resolution is not None:
            if line_opacity_mode == "c_k" and model_resolution > 1000:
                warnings.warn(
                    f"opacity resolving power ({model_resolution}) is above the maximum possible in c-k mode (1000), "
                    f"resetting the resolving power to its default value..."
                )
                self.model_resolution = None
            if line_opacity_mode == "lbl" and model_resolution < 1000:
                warnings.warn(
                    f"opacity resolving power ({model_resolution}) is below the maximum possible in c-k mode (1000), "
                    f"but current mode is 'lbl', which is inefficient\n"
                    f"To remove this warning, switch the line opacity mode to 'c-k'."
                )

        # Optional, covariance and scaling
        self.covariance: ArrayLike | None = None
        self.inv_cov: ArrayLike | None = None
        self.covariance_cholesky_factor: ArrayLike | None = None
        self.log_covariance_determinant: float | None = None
        self.fit_covariance: bool = fit_covariance
        self.covariance_mode: str = normalize_covariance_mode(covariance_mode)
        self.global_covariance_kernel: str = normalize_global_kernel_name(
            global_covariance_kernel
        )
        self.local_covariance_kernel: str = normalize_local_kernel_name(
            local_covariance_kernel
        )
        self.n_local_covariance_kernels: int = int(n_local_covariance_kernels)
        self.covariance_jitter: float = float(covariance_jitter)

        if covariance is not None:
            self.covariance = jnp.asarray(covariance, dtype=JAX_DTYPE)
            self._cache_covariance_factorization()

        if covariance is not None and uncertainties is None:
            self.uncertainties = jnp.sqrt(jnp.diagonal(self.covariance))

        self.scale_flux: bool = scale_flux
        self.scale_uncertainties: bool = scale_uncertainties
        self.fit_flux_offset: bool = fit_flux_offset
        self.fit_instrumental_resolution: bool = fit_instrumental_resolution
        self.subtract_continuum: bool = subtract_continuum
        self.scale_factor: float = 1.0
        self.offset: float = 0.0
        self.bval: float = -jnp.inf

        # Bins and photometry
        self.wavelength_boundaries: tuple[float, float] | None = None
        self.wavelength_bin_widths: ArrayLike | float | None = wavelength_bin_widths

        self.radtrans_grid = radtrans_grid

        # Read in data
        if path_to_observations is not None:
            # Check if data exists
            if not os.path.exists(path_to_observations):
                raise FileNotFoundError(
                    f"data file '{path_to_observations}' does not exist"
                )

            if path_to_observations.endswith("_x1d.fits"):
                self.load_x1d_fits(path_to_observations)
            elif path_to_observations.endswith(".fits"):
                self.loadfits(path_to_observations)
            else:
                self.loadtxt(path_to_observations)

        if self.wavelength_bin_widths is None:
            if wavelength_bin_widths is not None:
                self.wavelength_bin_widths = wavelength_bin_widths
            else:
                if self.wavelengths.ndim == 1:
                    self.wavelength_bin_widths = jnp.zeros_like(self.wavelengths)
                    self.wavelength_bin_widths = self.wavelength_bin_widths.at[
                        :-1
                    ].set(jnp.diff(self.wavelengths))
                    self.wavelength_bin_widths = self.wavelength_bin_widths.at[
                        -1
                    ].set(self.wavelength_bin_widths[-2])
                else:
                    self.wavelength_bin_widths = jnp.zeros_like(self.wavelengths)
                    self.wavelength_bin_widths = self.wavelength_bin_widths.at[:-1, :].set(
                        jnp.diff(self.wavelengths, axis=0)
                    )
                    self.wavelength_bin_widths = self.wavelength_bin_widths.at[-1, :].set(
                        self.wavelength_bin_widths[-2, :]
                    )

        self._validate_covariance_configuration()

        # Model wavelength boundaries.
        if wavelength_boundaries is not None:
            self.wavelength_boundaries = tuple(wavelength_boundaries)
        elif self.wavelengths is not None and jnp.size(jnp.asarray(self.wavelengths)) > 0:
            _wl = jnp.asarray(self.wavelengths)
            self.wavelength_boundaries = (
                0.98 * float(jnp.min(_wl)),
                1.02 * float(jnp.max(_wl)),
            )

    # TODO [4.0.0] This should be a classmethod instantiating a Data object, also fix function name => from_txt()
    def loadtxt(self, path: str, delimiter: str = ",", comments: str = "#") -> None:
        """
        Read a TXT or DAT file containing a header above 3 or 4 data columns representing a spectrum.
        Headers should start with the symbol given in the 'comment' argument.
        The 4 possible data columns are, from left to right:
            1. The wavelengths in microns,
            2. [Optional] The wavelengths bin widths, in microns.
            3. The flux or transit depths,
            4. The error on each data point.

        Alternatively, the wavelength column
        Checks will be performed to determine the correct delimiter, but the recommended format is to use a CSV file
        with columns for wavelength, flux and error.

        Args:
            path : str
                Directory and filename of the data.
            delimiter : string, int
                The string used to separate values. By default, commas act as delimiter.
                An integer or sequence of integers can also be provided as width(s) of each field.
            comments : string
                The character used to indicate the start of a comment.
                All the characters occurring on a line after a comment are discarded
        """
        if self.__class__ == PhotometryData:
            return

        obs = np.genfromtxt(path, delimiter=delimiter, comments=comments)
        obs = jnp.array(obs)
        # Input sanity checks
        if jnp.isnan(obs).any():
            obs = jnp.array(genfromtxt(path))
        if len(obs.shape) < 2:
            obs = jnp.array(genfromtxt(path, comments=comments))
        if obs.shape[1] == 4:
            self.wavelengths = obs[:, 0]
            self.wavelength_bin_widths = obs[:, 1]
            self.spectrum = obs[:, 2]
            self.uncertainties = obs[:, 3]
            return
        elif obs.shape[1] != 3:
            obs = jnp.array(genfromtxt(path))

        # Warnings and errors
        if obs.shape[1] < 3:
            raise ValueError(
                f"obs file '{path}' must contain at least 3 columns (wavelength, flux, flux error), "
                f"but has {obs.shape[1]} column(s)"
            )
        elif obs.shape[1] > 4:
            warnings.warn(
                f"obs file '{path}' should contain at most 4 columns "
                f"(wavelength, wavelength bins (optional), flux, flux error), "
                f"but has {obs.shape[1]} column(s)\n"
                f"Additional columns will be ignored"
            )

        if jnp.isnan(obs).any():
            warnings.warn(f"NANs present in obs file '{path}'")

        self.wavelengths = obs[:, 0]
        self.spectrum = obs[:, 1]
        self.uncertainties = obs[:, 2]

    def load_x1d_fits(self, path):
        # TODO [4.0.0] This should be a classmethod instantiating a Data object, named "from_jwst_data"
        """
        Load in a x1d fits file as produced by the STSci JWST pipeline.
        Expects units of Jy for the flux and micron for the wavelength.

        Args:
            path : str
                Directory and filename of the data.
        """
        hdul = fits.open(path)
        self.wavelengths = hdul["EXTRACT1D"].data["WAVELENGTH"]
        self.spectrum = hdul["EXTRACT1D"].data["FLUX"]
        self.uncertainties = hdul["EXTRACT1D"].data["FLUX_ERROR"]

        # Convert from Jy to W/m^2/micron
        c_micron = cst.c * 1e4
        self.spectrum = 1e-26 * c_micron * self.spectrum / self.wavelengths**2
        self.uncertainties = 1e-26 * c_micron * self.uncertainties / self.wavelengths**2

    # TODO [4.0.0] This should be a classmethod instantiating a Data object, named "from_fits"
    def loadfits(self,
                 path,
                 wavelength_column_name="WAVELENGTH",
                 flux_column_name="FLUX",
                 error_column_name="ERROR",
                 covariance_column_name="COVARIANCE") -> None:
        """
        Load a Radtrans-formatted fits file.
        Must include extension SPECTRUM with fields WAVELENGTH, FLUX
        and COVARIANCE (or ERROR).

        Args:
            path : str
                Directory and filename of the data.
        """

        data = fits.getdata(path, "SPECTRUM")

        if hasattr(data, "columns") and data.columns is not None:
            column_names = tuple(data.columns.names)
            self.wavelengths = jnp.asarray(data.field(wavelength_column_name), dtype=JAX_DTYPE)
            self.spectrum = jnp.asarray(data.field(flux_column_name), dtype=JAX_DTYPE)

            if covariance_column_name in column_names:
                self.covariance = jnp.asarray(data.field(covariance_column_name), dtype=JAX_DTYPE)
                self._cache_covariance_factorization()
                self.uncertainties = jnp.sqrt(self.covariance.diagonal())
            elif error_column_name in column_names:
                # Note that this will only be the uncorrelated error.
                # Dot with the correlation matrix (if available) to get
                # the full error.
                self.uncertainties = jnp.asarray(data.field(error_column_name), dtype=JAX_DTYPE)
                self.covariance = None
                self.inv_cov = None
                self.covariance_cholesky_factor = None
                self.log_covariance_determinant = None
        else:
            hdul = fits.open(path)
            self.wavelengths = jnp.asarray(hdul[1].data[wavelength_column_name], dtype=JAX_DTYPE)
            self.spectrum = jnp.asarray(hdul[1].data[flux_column_name], dtype=JAX_DTYPE)

            if error_column_name in hdul[1].data.columns.names:
                self.uncertainties = jnp.asarray(
                    hdul[1].data[error_column_name], dtype=JAX_DTYPE
                )

            if covariance_column_name in hdul[1].data.columns.names:
                self.covariance = jnp.asarray(hdul[1].data[covariance_column_name], dtype=JAX_DTYPE)

            if self.covariance is not None:
                self._cache_covariance_factorization()

        if self.uncertainties is None and self.covariance is not None:
            self.uncertainties = jnp.sqrt(self.covariance.diagonal())

    def initialise_data_resolution(self, wavelengths_model: ArrayLike) -> None:
        # TODO [4.0.0] this function should raise an error when the interpolation is not possible
        #  As it is now, it can do nothing, which can be confusing
        if isinstance(self.data_resolution, (jnp.ndarray, np.ndarray)):
            self.data_resolution_array_model = jnp.interp(
                wavelengths_model, self.wavelengths, jnp.asarray(self.data_resolution)
            )

    def update_bins(self, wavelengths: ArrayLike) -> None:
        """
        Update the wavelength bin widths based on the provided wavelength array.

        This method computes the bin width for each wavelength bin as the difference
        between consecutive wavelength values. The last bin width is set equal to the
        second-to-last bin width to maintain array length consistency.

        Args:
            wavelengths : ArrayLike
                Array of wavelength bin centers or edges.
        """
        self.wavelength_bin_widths = jnp.zeros_like(wavelengths)
        self.wavelength_bin_widths[:-1] = jnp.diff(wavelengths)
        self.wavelength_bin_widths[-1] = self.wavelength_bin_widths[-2]

    def line_b_uncertainty_scaling(self, parameters):
        """
        This function implements the 10^b scaling from Line 2015, which allows
        for us to account for underestimated uncertainties:

        We modify the standard error on the data point by the factor 10^b to account for
        underestimated uncertainties and/or unknown missing forward model physics
        (Foreman-Mackey et al. 2013, Hogg et al. 2010, Tremain et al. 2002), e.g., imperfect fits.
        This results in a more generous estimate of the parameter uncertainties. Note that this
        is similar to inflating the error bars post-facto in order to achieve reduced chi-squares
        of unity, except that this approach is more formal because uncertainties in this parameter
        are properly marginalized into the other relevant parameters. Generally, the factor 10^b
        takes on values that fall between the minimum and maximum of the square of the data uncertainties.

        Args:
            parameters: Dict
                Dictionary of Parameters, should contain key 'uncertianty_scaling_b'.
                This can be done for all data sets, or specified with a tag at the end of
                the key to apply different factors to different datasets.
        Returns:
            b: float
                10**b error bar scaling factor.
        """
        return _line_b_uncertainty_scaling(self.name, parameters)

    def _validate_covariance_configuration(self) -> None:
        if not self.fit_covariance:
            return

        if self.covariance_mode == "none":
            self.fit_covariance = False
            self.n_local_covariance_kernels = 0
            return

        if getattr(self.spectrum, "dtype", None) == "O":
            raise NotImplementedError(
                f"Covariance fitting is currently unsupported for ragged dataset '{self.name}'"
            )

        if self.wavelengths is not None and jnp.ndim(self.wavelengths) != 1:
            raise NotImplementedError(
                f"Covariance fitting requires 1D wavelength grids for dataset '{self.name}'"
            )

        if self.spectrum is not None and jnp.ndim(self.spectrum) != 1:
            raise NotImplementedError(
                f"Covariance fitting requires 1D spectra for dataset '{self.name}'"
            )

        if self.covariance_mode == "global":
            self.n_local_covariance_kernels = 0
        elif (
            self.covariance_mode in {"local", "global_local"}
            and self.n_local_covariance_kernels < 1
        ):
            self.n_local_covariance_kernels = 1

    def to_observation_state(self, model_contract: str = "legacy") -> ObservationState:
        return ObservationState(
            name=self.name,
            wavelengths=self.wavelengths,
            spectrum=self.spectrum,
            mask=self.mask,
            uncertainties=self.uncertainties,
            covariance=self.covariance,
            radtrans_object=self.radtrans_object,
            model_generating_function=self.model_generating_function,
            model_resolution=self.model_resolution,
            data_resolution=self.data_resolution,
            data_resolution_array_model=self.data_resolution_array_model,
            wavelength_bin_widths=self.wavelength_bin_widths,
            external_radtrans_reference=self.external_radtrans_reference,
            line_opacity_mode=self.line_opacity_mode,
            model_contract=model_contract,
            scale_flux=self.scale_flux,
            scale_uncertainties=self.scale_uncertainties,
            fit_flux_offset=self.fit_flux_offset,
            fit_instrumental_resolution=self.fit_instrumental_resolution,
            subtract_continuum=self.subtract_continuum,
            fit_covariance=self.fit_covariance,
            covariance_mode=self.covariance_mode,
            global_covariance_kernel=self.global_covariance_kernel,
            local_covariance_kernel=self.local_covariance_kernel,
            n_local_covariance_kernels=self.n_local_covariance_kernels,
            covariance_jitter=self.covariance_jitter,
            metadata={
                "radtrans_grid": self.radtrans_grid,
                "wavelength_boundaries": self.wavelength_boundaries,
            },
        )

    def _cache_covariance_factorization(self) -> None:
        self.inv_cov = None
        self.covariance_cholesky_factor = None
        self.log_covariance_determinant = None

        if self.covariance is None:
            return

        covariance_np = np.asarray(self.covariance)
        cholesky_factor = np.linalg.cholesky(covariance_np)

        self.covariance_cholesky_factor = jnp.asarray(cholesky_factor, dtype=JAX_DTYPE)
        self.log_covariance_determinant = self.covariance.shape[0] * jnp.log(
            2.0 * jnp.pi
        ) + 2.0 * jnp.sum(jnp.log(jnp.diagonal(self.covariance_cholesky_factor)))

    def refresh_covariance_bookkeeping(self) -> None:
        self._cache_covariance_factorization()

        if self.covariance is not None and self.uncertainties is None:
            self.uncertainties = jnp.sqrt(jnp.diagonal(self.covariance))

        self._validate_covariance_configuration()

    def to_likelihood_state(
        self,
        n_free_parameters: int = 0,
        parameter_names: tuple[str, ...] = (),
    ) -> LikelihoodState:
        parameter_names = set(parameter_names)
        has_parameter_dependent_noise = (
            self.fit_covariance
            or self.scale_uncertainties
            or f"{self.name}_b" in parameter_names
            or "uncertainty_scaling_b" in parameter_names
        )

        data_covariance = self.covariance if has_parameter_dependent_noise else None
        covariance_cholesky_factor = (
            self.covariance_cholesky_factor
            if self.covariance is not None and not has_parameter_dependent_noise
            else None
        )
        log_likelihood_normalization = None
        inv_covariance_diagonal = None

        if not has_parameter_dependent_noise:
            if covariance_cholesky_factor is not None:
                log_likelihood_normalization = -0.5 * self.log_covariance_determinant
                # Pre-computed once for the fixed-covariance case so the
                # per-datapoint (PSIS-LOO) path avoids a triangular solve
                # against the identity on every sample.
                inverse_cholesky_factor = solve_triangular(
                    covariance_cholesky_factor,
                    jnp.eye(covariance_cholesky_factor.shape[0], dtype=JAX_DTYPE),
                    lower=True,
                )
                inv_covariance_diagonal = jnp.sum(inverse_cholesky_factor**2, axis=0)
            elif self.uncertainties is not None:
                log_likelihood_normalization = -0.5 * jnp.sum(
                    jnp.log(2.0 * jnp.pi * self.uncertainties**2)
                )

        return LikelihoodState(
            data_uncertainties=self.uncertainties,
            data_covariance=data_covariance,
            mask=self.mask,
            inv_covariance=None,
            inv_covariance_diagonal=inv_covariance_diagonal,
            covariance_cholesky_factor=covariance_cholesky_factor,
            log_covariance_determinant=self.log_covariance_determinant,
            log_likelihood_normalization=log_likelihood_normalization,
            n_free_parameters=n_free_parameters,
            metadata={
                "name": self.name,
                "has_covariance": self.covariance is not None,
                "fit_covariance": self.fit_covariance,
                "covariance_mode": self.covariance_mode,
                "has_parameter_dependent_noise": has_parameter_dependent_noise,
                "data_resolution": self.data_resolution,
            },
        )

    def apply_flux_scaling(self, input_spectrum, parameters_dict: dict) -> ArrayLike:
        """Applies multiplicative flux scaling based on parameters in parameters_dict."""
        return _apply_flux_scaling(
            input_spectrum, self.to_observation_state(), parameters_dict
        )

    def offset_flux(self, input_spectrum, parameters_dict: dict) -> ArrayLike:
        """Applies additive flux offset based on parameters in parameters_dict."""
        return _offset_flux(
            input_spectrum, self.to_observation_state(), parameters_dict
        )

    def scale_and_inflate_uncertainties(
        self, parameters_dict: dict
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Applies scaling to self.uncertainties or self.covariance.
        Returns a tuple: (scaled_uncertainties, scaled_covariance).
        """
        return _scale_and_inflate_uncertainties(
            self.to_observation_state(), parameters_dict
        )


class PhotometryData(Data):
    """
    Data class for photometric observations.
    Handles single or multiple photometric bands.
    """

    def __init__(
        self,
        name,
        photometric_transformation_function,
        model_generating_function,
        photometric_bin_edges=None,
        wavelength_boundaries=None,
        path_to_observations=None,
        data_resolution=None,
        model_resolution=None,
        external_radtrans_reference=None,
        scale_flux=False,
        scale_uncertainties=False,
        fit_flux_offset=False,
        line_opacity_mode="c-k",
        radtrans_object=None,
        wavelengths=None,
    ):

        self.photometric_transformation_function = photometric_transformation_function

        if self.photometric_transformation_function is None:
            raise ValueError(
                f"{name}: photometric_transformation_function must be provided for PhotometryData."
            )

        # Edges of the data
        if photometric_bin_edges is None:
            raise ValueError(
                f"{name}: photometric_bin_edges must be provided for PhotometryData."
            )

        self.photometric_bin_edges = jnp.asarray(photometric_bin_edges)

        # Handle shape of photometric_bin_edges (single band vs multiple bands)
        if self.photometric_bin_edges.ndim == 1:  # Single band [low, high]
            if self.photometric_bin_edges.shape != (2,):
                raise ValueError(
                    f"{name}: photometric_bin_edges for a single band must be a pair [low, high]."
                )
        elif (
            self.photometric_bin_edges.ndim == 2
        ):  # Multiple bands [[low1, high1], [low2, high2]]
            if self.photometric_bin_edges.shape[1] != 2:
                raise ValueError(
                    f"{name}: photometric_bin_edges for multiple bands must be an array of pairs [[low, high], ...]."
                )
        else:
            raise ValueError(f"{name}: photometric_bin_edges has invalid shape.")

        # Edges of the model
        # Determine wavelength_boundaries for the base RadTrans object
        # If not provided, derive from the full range of photometric bins.
        derived_wavelength_boundaries = wavelength_boundaries
        if derived_wavelength_boundaries is None:
            if self.photometric_bin_edges.ndim == 1:  # Single band
                derived_wavelength_boundaries = [
                    0.98 * self.photometric_bin_edges[0],
                    1.02 * self.photometric_bin_edges[1],
                ]
            else:  # Multiple bands
                derived_wavelength_boundaries = [
                    0.98 * jnp.min(self.photometric_bin_edges[:, 0]),
                    1.02 * jnp.max(self.photometric_bin_edges[:, 1]),
                ]

        # Calculate effective/central wavelengths and bin widths for base class
        # These will be stored as self.wavelengths and self.wavelength_bin_widths by the base class
        calculated_effective_wavelengths = wavelengths
        calculated_wavelength_bin_widths = None  # To be passed to base

        if self.photometric_bin_edges.ndim == 1:  # Single band
            if calculated_effective_wavelengths is None:
                calculated_effective_wavelengths = jnp.mean(self.photometric_bin_edges)
            calculated_wavelength_bin_widths = (
                self.photometric_bin_edges[1] - self.photometric_bin_edges[0]
            )
        else:  # Multiple bands
            if calculated_effective_wavelengths is None:
                calculated_effective_wavelengths = jnp.mean(
                    self.photometric_bin_edges, axis=1
                )
            calculated_wavelength_bin_widths = (
                self.photometric_bin_edges[:, 1] - self.photometric_bin_edges[:, 0]
            )

        # Ensure calculated_effective_wavelengths is a 1D array for consistency
        if calculated_effective_wavelengths is not None:
            calculated_effective_wavelengths = jnp.asarray(
                [calculated_effective_wavelengths]
            ).flatten()
        if calculated_wavelength_bin_widths is not None:
            calculated_wavelength_bin_widths = jnp.asarray(
                [calculated_wavelength_bin_widths]
            ).flatten()

        super().__init__(
            name=name,
            path_to_observations=path_to_observations,
            data_resolution=data_resolution,
            model_resolution=model_resolution,
            external_radtrans_reference=external_radtrans_reference,
            model_generating_function=model_generating_function,
            wavelength_boundaries=derived_wavelength_boundaries,
            scale_flux=scale_flux,
            scale_uncertainties=scale_uncertainties,
            fit_flux_offset=fit_flux_offset,
            wavelength_bin_widths=calculated_wavelength_bin_widths,
            line_opacity_mode=line_opacity_mode,
            radtrans_object=radtrans_object,
            wavelengths=calculated_effective_wavelengths,
        )

        self.wavelength_boundaries = derived_wavelength_boundaries
        self.wavelength_bin_widths = calculated_wavelength_bin_widths

        if self.data_resolution is None:
            inferred_data_resolution = jnp.asarray(
                calculated_effective_wavelengths / calculated_wavelength_bin_widths
            )
            self.data_resolution = (
                float(inferred_data_resolution[0])
                if inferred_data_resolution.size == 1
                else inferred_data_resolution
            )

    def _build_photometry_metadata(self) -> dict[str, object]:
        return {
            "photometry": True,
            "photometric_bin_edges": self.photometric_bin_edges,
            "photometric_transformation_function_name": getattr(
                self.photometric_transformation_function,
                "__name__",
                type(self.photometric_transformation_function).__name__,
            ),
        }

    def to_observation_state(self, model_contract: str = "legacy") -> ObservationState:
        obs = super().to_observation_state(model_contract=model_contract)
        updated_metadata = dict(obs.metadata)
        updated_metadata.update(self._build_photometry_metadata())
        return PhotometryObservationState(
            name=obs.name,
            wavelengths=obs.wavelengths,
            spectrum=obs.spectrum,
            mask=obs.mask,
            uncertainties=obs.uncertainties,
            covariance=obs.covariance,
            radtrans_object=obs.radtrans_object,
            model_generating_function=obs.model_generating_function,
            model_resolution=obs.model_resolution,
            data_resolution=obs.data_resolution,
            data_resolution_array_model=obs.data_resolution_array_model,
            wavelength_bin_widths=obs.wavelength_bin_widths,
            external_radtrans_reference=obs.external_radtrans_reference,
            line_opacity_mode=obs.line_opacity_mode,
            model_contract=obs.model_contract,
            scale_flux=obs.scale_flux,
            scale_uncertainties=obs.scale_uncertainties,
            fit_flux_offset=obs.fit_flux_offset,
            fit_instrumental_resolution=obs.fit_instrumental_resolution,
            subtract_continuum=obs.subtract_continuum,
            photometric_transformation_function=(
                self.photometric_transformation_function
            ),
            fit_covariance=obs.fit_covariance,
            covariance_mode=obs.covariance_mode,
            global_covariance_kernel=obs.global_covariance_kernel,
            local_covariance_kernel=obs.local_covariance_kernel,
            n_local_covariance_kernels=obs.n_local_covariance_kernels,
            covariance_jitter=obs.covariance_jitter,
            metadata=updated_metadata,
        )

    def to_likelihood_state(
        self,
        n_free_parameters: int = 0,
        parameter_names: tuple[str, ...] = (),
    ) -> LikelihoodState:
        likelihood = super().to_likelihood_state(
            n_free_parameters=n_free_parameters,
            parameter_names=parameter_names,
        )
        updated_metadata = dict(likelihood.metadata)
        updated_metadata.update(self._build_photometry_metadata())
        return LikelihoodState(
            data_uncertainties=likelihood.data_uncertainties,
            data_covariance=likelihood.data_covariance,
            mask=likelihood.mask,
            inv_covariance=likelihood.inv_covariance,
            inv_covariance_diagonal=likelihood.inv_covariance_diagonal,
            covariance_cholesky_factor=likelihood.covariance_cholesky_factor,
            log_covariance_determinant=likelihood.log_covariance_determinant,
            log_likelihood_normalization=likelihood.log_likelihood_normalization,
            n_free_parameters=likelihood.n_free_parameters,
            metadata=updated_metadata,
        )


class TimeSeriesData(Data):
    """
    Data class for photometric observations.
    Handles single or multiple photometric bands.
    """

    def __init__(
        self,
        name,
        path_to_observations=None,
        filename_list=None,
        observation_times=None,
        data_resolution=None,
        model_resolution=None,
        external_radtrans_reference=None,
        model_generating_function=None,
        wavelength_boundaries=None,
        scale_flux=False,
        scale_uncertainties=False,
        fit_flux_offset=False,
        line_opacity_mode="c-k",
        radtrans_grid=False,
        concatenate_flux_epochs_variability=False,
        atmospheric_column_flux_mixer=None,
        variability_atmospheric_column_model_flux_return_mode=False,
        radtrans_object=None,
        spectrum=None,
        wavelengths=None,
        wavelength_bin_widths=None,
        uncertainties=None,
        covariance=None,
        mask=None,
    ):
        if observation_times is not None:
            self.observation_times = jnp.asarray(observation_times, dtype=JAX_DTYPE)
        else:
            self.observation_times = None
        self.filename_list = filename_list
        self.concatenate_flux_epochs_variability = concatenate_flux_epochs_variability
        self.atmospheric_column_flux_mixer = atmospheric_column_flux_mixer
        self.variability_atmospheric_column_model_flux_return_mode = (
            variability_atmospheric_column_model_flux_return_mode
        )
        # Time-series model configuration (set by RetrievalConfig.add_time_series_data)
        self.time_series_config = None
        super().__init__(
            name=name,
            path_to_observations=path_to_observations,
            data_resolution=data_resolution,
            model_resolution=model_resolution,
            external_radtrans_reference=external_radtrans_reference,
            model_generating_function=model_generating_function,
            wavelength_boundaries=wavelength_boundaries,
            scale_flux=scale_flux,
            scale_uncertainties=scale_uncertainties,
            fit_flux_offset=fit_flux_offset,
            line_opacity_mode=line_opacity_mode,
            radtrans_grid=radtrans_grid,
            radtrans_object=radtrans_object,
            spectrum=spectrum,
            wavelengths=wavelengths,
            wavelength_bin_widths=wavelength_bin_widths,
            uncertainties=uncertainties,
            covariance=covariance,
            mask=mask,
        )

    @staticmethod
    def _collapse_shared_epoch_axis(values):
        if values is None:
            return None

        values_array = np.asarray(values)

        if values_array.ndim < 2:
            return values

        reference_row = values_array[0]

        if np.allclose(values_array, reference_row[None, ...], equal_nan=True):
            return values[0]

        return values

    def to_observation_state(self, model_contract: str = "legacy") -> ObservationState:
        obs = super().to_observation_state(model_contract=model_contract)
        updated_metadata = dict(obs.metadata)
        updated_metadata["observation_times"] = self.observation_times
        if self.time_series_config is not None:
            updated_metadata["time_series_config"] = self.time_series_config

        observation_wavelengths = self._collapse_shared_epoch_axis(obs.wavelengths)
        observation_bin_widths = self._collapse_shared_epoch_axis(
            obs.wavelength_bin_widths
        )
        observation_data_resolution = self._collapse_shared_epoch_axis(
            obs.data_resolution
        )
        observation_data_resolution_array_model = self._collapse_shared_epoch_axis(
            obs.data_resolution_array_model
        )

        return ObservationState(
            name=obs.name,
            wavelengths=observation_wavelengths,
            spectrum=obs.spectrum,
            mask=obs.mask,
            uncertainties=obs.uncertainties,
            covariance=obs.covariance,
            radtrans_object=obs.radtrans_object,
            model_generating_function=obs.model_generating_function,
            model_resolution=obs.model_resolution,
            data_resolution=observation_data_resolution,
            data_resolution_array_model=observation_data_resolution_array_model,
            wavelength_bin_widths=observation_bin_widths,
            external_radtrans_reference=obs.external_radtrans_reference,
            line_opacity_mode=obs.line_opacity_mode,
            model_contract=obs.model_contract,
            scale_flux=obs.scale_flux,
            scale_uncertainties=obs.scale_uncertainties,
            fit_flux_offset=obs.fit_flux_offset,
            fit_instrumental_resolution=obs.fit_instrumental_resolution,
            subtract_continuum=obs.subtract_continuum,
            photometric_transformation_function=obs.photometric_transformation_function,
            fit_covariance=obs.fit_covariance,
            covariance_mode=obs.covariance_mode,
            global_covariance_kernel=obs.global_covariance_kernel,
            local_covariance_kernel=obs.local_covariance_kernel,
            n_local_covariance_kernels=obs.n_local_covariance_kernels,
            covariance_jitter=obs.covariance_jitter,
            variability_atmospheric_column_model_flux_return_mode=(
                obs.variability_atmospheric_column_model_flux_return_mode
            ),
            metadata=updated_metadata,
        )

    def load_single_spectrum_txt(self, delimiter=",", comments="#"):
        wavelengths, spectra, bin_widths, uncertainties = [], [], [], []
        for filename in self.filename_list:
            wavelength, spectrum, bin_width, uncertainty = self.loadtxt(
                self.path_to_observations + filename, delimiter, comments
            )
            wavelengths.append(wavelength)
            spectra.append(spectrum)
            bin_widths.append(bin_width)
            uncertainties.append(uncertainty)
        self.wavelengths = jnp.array(wavelengths)
        self.spectrum = jnp.array(spectra)
        self.wavelength_bin_widths = jnp.array(bin_widths)
        self.uncertainties = jnp.array(uncertainties)

    def loadtxt(self, path, delimiter=",", comments="#"):
        """
        This function reads in a .txt or .dat file containing the spectrum. Headers should be commented out with '#',
        the first column must be the wavelength in micron, the second column the flux or transit depth,
        and the final column must be the error on each data point.
        Checks will be performed to determine the correct delimiter, but the recommended format is to use a
        csv file with columns for wavelength, flux and error.

        Args:
            path : str
                Directory and filename of the data.
            delimiter : string, int
                The string used to separate values. By default, commas act as delimiter.
                An integer or sequence of integers can also be provided as width(s) of each field.
            comments : string
                The character used to indicate the start of a comment.
                All the characters occurring on a line after a comment are discarded
        """
        obs = jnp.array(genfromtxt(path, delimiter=delimiter, comments=comments))
        # Input sanity checks
        if jnp.isnan(obs).any():
            obs = jnp.array(genfromtxt(path))
        if len(obs.shape) < 2:
            obs = jnp.array(genfromtxt(path, comments=comments))
        if obs.shape[1] == 4:
            wavelengths = obs[:, 0]
            wavelength_bin_widths = obs[:, 1]
            spectrum = obs[:, 2]
            uncertainties = obs[:, 3]
            return wavelengths, wavelength_bin_widths, spectrum, uncertainties
        elif obs.shape[1] != 3:
            obs = jnp.array(genfromtxt(path))

        # Warnings and errors
        if obs.shape[1] < 3:
            raise ValueError(
                f"data file '{path}' must contain at least 3 columns (wavelength, flux, flux error), "
                f"but has {obs.shape[1]}"
            )
        elif obs.shape[1] > 4:
            warnings.warn(
                f"data file '{path}' should contain at most 4 columns "
                f"(wavelength, [opt, wavelength bins], flux, flux error), "
                f"but has {obs.shape[1]}\n"
                f"Additional columns will be ignored"
            )

        if jnp.isnan(obs).any():
            warnings.warn(f"NANs present in data file '{path}'")

        wavelengths = obs[:, 0]
        spectrum = obs[:, 1]
        uncertainties = obs[:, 2]
        wavelength_bin_widths = jnp.zeros_like(wavelengths)
        wavelength_bin_widths = wavelength_bin_widths.at[:-1].set(jnp.diff(wavelengths))
        wavelength_bin_widths = wavelength_bin_widths.at[-1].set(
            wavelength_bin_widths[-2]
        )
        return wavelengths, wavelength_bin_widths, spectrum, uncertainties

    def initialise_data_resolution(self, wavelengths_model: ArrayLike) -> None:
        if isinstance(self.data_resolution, (jnp.ndarray, np.ndarray)):
            shared_wavelengths = self._collapse_shared_epoch_axis(self.wavelengths)
            shared_resolution = self._collapse_shared_epoch_axis(self.data_resolution)

            shared_wavelengths = jnp.asarray(shared_wavelengths)
            shared_resolution = jnp.asarray(shared_resolution)

            if shared_wavelengths.ndim != 1 or shared_resolution.ndim != 1:
                raise ValueError(
                    "TimeSeriesData variable spectral resolution requires a shared 1-D wavelength grid"
                )

            if shared_wavelengths.shape[0] != shared_resolution.shape[0]:
                raise ValueError(
                    "TimeSeriesData wavelengths and spectral resolution must have matching lengths"
                )

            self.data_resolution_array_model = jnp.interp(
                wavelengths_model, shared_wavelengths, shared_resolution
            )
        else:
            raise ValueError("data_resolution must be array like to interpolate!")
