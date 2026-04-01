import copy
import os
import warnings
from typing import Callable

import numpy as np
import jax.numpy as jnp
from jax import config
from numpy import genfromtxt
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
    _apply_flux_scaling,
    _offset_flux,
    _line_b_uncertainty_scaling,
    _scale_and_inflate_uncertainties,
)

JAX_DTYPE = jnp.float32
if config.jax_enable_x64:
    JAX_DTYPE = jnp.float64


class Data:
    r"""
    This class stores the spectral data to be retrieved from a single instrument or observation.

    Each dataset is associated with an instance of petitRadTrans and an atmospheric model.
    The pRT instance can be overwritten, and associated with an existing pRT instance with the
    external_pRT_reference parameter.
    This setup allows for joint or independent retrievals on multiple datasets.
    # TODO @Evert complete docstring
    Args:
        name : str
            Identifier for this data set.
        path_to_observations : str
            Path to observations file, including filename. This can be a txt or dat file
            containing the wavelength, flux, transit depth and error, or a fits file
            containing the wavelength, spectrum and covariance matrix.
            Alternatively, the data information can be directly given by the wavelengths, spectrum, uncertainties, and
            mask attributes.
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
        system_distance : float
            The distance to the object in CGS units. Defaults to a 10pc normalized distance.
        external_radtrans_reference : object
            An existing RadTrans object. Leave as none unless you're sure of what you're doing.
        model_generating_function : method
            A function, typically defined in run_definition.py that returns the model wavelength and spectrum
            (emission or transmission).
            This is the function that contains the physics of the model, and calls pRT in order to compute the
            spectrum.
        wavelength_boundaries : tuple,list
            Set the wavelength range of the pRT object. Defaults to a range +/-5% greater than that of the data.
            Must at least be equal to the range of the data.
        scale_flux : bool
            Turn on or off scaling the data by a constant factor. Set to True if scaling the data during the
            retrieval.
        scale_uncertainties:
            # TODO @Evert complete docstring
        fit_flux_offset:
            # TODO @Evert complete docstring
        wavelength_bin_widths : numpy.ndarray
            Set the wavelength bin width to bin the Radtrans object to the data. Defaults to the data bins.
        photometry : bool
            Set to True if using photometric data.
        photometric_transformation_function : method
            Transform the photometry (account for filter transmission etc.).
            This function must take in the wavelength and flux of a spectrum,
            and output a single photometric point (and optionally flux error).
        photometric_bin_edges : Tuple, numpy.ndarray
            The edges of the photometric bin in micron. [low,high]
        line_opacity_mode : str
            Should the retrieval be run using correlated-k opacities (default, 'c-k'),
            or line by line ('lbl') opacities? If 'lbl' is selected, it is HIGHLY
            recommended to set the model_resolution parameter. In general,
            'c-k' mode is recommended for retrievals of everything other than
            high-resolution (R>40000) spectra.
        radtrans_grid: bool
            Set to true if data has been binned to a pRT c-k grid.
        concatenate_flux_epochs_variability: bool
            Set to true if data concatenation treatment for variability is to be used.
        atmospheric_column_flux_mixer: method
            Function that mixes model fluxes of atmospheric columns in variability retrievals.
        variability_atmospheric_column_model_flux_return_mode: bool
            Set to true if the forward model should returns the fluxes of the individual atmospheric
            columns. This is useful if external_radtrans_reference is True, but the master (reference) object
            should return the column fluxes for mixing, not the combined column flux. In this case a column
            mixing function needs to be handed to the data constructor.
        radtrans_object:
            An instance of Radtrans object to be used to generate model spectra in retrievals.
        wavelengths:
            (um) Wavelengths of the data.
        spectrum:
            Spectrum of the data.
        uncertainties:
            Uncertainties of the data, in the same units as the spectrum.
        covariance:
            Covariance matrix of the data, in the same units as the spectrum squared.
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
        mask:
            Mask of the data.
    """

    resolving_power_str = ".R"

    def __init__(
        self,
        name: str,
        path_to_observations: str | None = None,
        data_resolution: float | None = None,
        model_resolution: float | None = None,
        system_distance: float | None = None,
        external_radtrans_reference: object | None = None,
        model_generating_function: (
            Callable[..., tuple[ArrayLike, ArrayLike, ArrayLike]] | None
        ) = None,
        wavelength_boundaries: tuple[float, float] | None = None,
        scale_flux: bool = False,
        scale_uncertainties: bool = False,
        fit_flux_offset: bool = False,
        fit_instrumental_resolution: bool = False,
        subtract_continuum: bool = False,
        wavelength_bin_widths: ArrayLike | None = None,
        photometry: bool = False,
        photometric_transformation_function: (
            Callable[[ArrayLike, ArrayLike], tuple[float, float]] | None
        ) = None,
        photometric_bin_edges: tuple[float, float] | None = None,
        line_opacity_mode: str = "c-k",
        radtrans_grid: bool = False,
        concatenate_flux_epochs_variability: bool = False,
        atmospheric_column_flux_mixer: Callable | None = None,
        variability_atmospheric_column_model_flux_return_mode: bool = False,
        radtrans_object: object = None,
        wavelengths: ArrayLike | None = None,
        spectrum: ArrayLike | None = None,
        uncertainties: ArrayLike | None = None,
        covariance: ArrayLike | None = None,
        fit_covariance: bool = False,
        covariance_mode: str = "none",
        global_covariance_kernel: str = "squared_exponential",
        local_covariance_kernel: str = "squared_exponential",
        n_local_covariance_kernels: int = 0,
        covariance_jitter: float = 0.0,
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

        self.system_distance: float

        if system_distance is None:
            self.system_distance = 10 * cst.pc
        else:
            self.system_distance = system_distance

        # Sanity check system distance
        if self.system_distance < cst.pc:
            warnings.warn(
                f"system distance ({self.system_distance}) is less than 1 pc, make sure that its units are in CGS"
            )

        self.data_resolution: float | None = data_resolution
        self.data_resolution_array_model: ArrayLike | None = None

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
        self.covariance: ArrayLike | None = covariance
        self.inv_cov: ArrayLike | None = None
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
            self.inv_cov = jnp.linalg.inv(covariance)
            sign, self.log_covariance_determinant = jnp.linalg.slogdet(
                2.0 * jnp.pi * covariance
            )

        if covariance is not None and uncertainties is None:
            self.uncertainties = jnp.sqrt(jnp.diagonal(covariance))

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
        self.photometry = photometry
        self.photometric_transformation_function = photometric_transformation_function

        if photometry:
            missing = []

            if photometric_transformation_function is None:
                missing.append("'photometric_transformation_function'")

            if photometric_bin_edges is None:
                missing.append("'photometric_bin_edges'")

            if len(missing) > 0:
                ", ".join(missing)
                raise ValueError(
                    f"missing photometric arguments for photometric data '{name}': {missing}"
                )

        self.photometry_range = wavelength_boundaries
        self.photometric_bin_edges: tuple[float, float] | None = photometric_bin_edges

        self.radtrans_grid = radtrans_grid
        self.concatenate_flux_epochs_variability = concatenate_flux_epochs_variability
        self.variability_atmospheric_column_model_flux_return_mode = (
            variability_atmospheric_column_model_flux_return_mode
        )
        self.atmospheric_column_flux_mixer = atmospheric_column_flux_mixer

        # Read in data
        if path_to_observations is not None:
            # Check if data exists
            if not os.path.exists(path_to_observations):
                raise FileNotFoundError(
                    f"data file '{path_to_observations}' does not exist"
                )

            if not photometry:
                if path_to_observations.endswith("_x1d.fits"):
                    self.load_x1d_fits(path_to_observations)
                elif path_to_observations.endswith(".fits"):
                    self.loadfits(path_to_observations)
                else:
                    self.loadtxt(path_to_observations)

                if wavelength_boundaries is not None:
                    self.wavelength_boundaries = wavelength_boundaries
                else:
                    self.wavelength_boundaries = (
                        0.95 * self.wavelengths[0],
                        1.05 * self.wavelengths[-1],
                    )

                if self.wavelength_bin_widths is None:
                    if wavelength_bin_widths is not None:
                        self.wavelength_bin_widths = wavelength_bin_widths
                    else:
                        self.wavelength_bin_widths = jnp.zeros_like(self.wavelengths)
                        self.wavelength_bin_widths = self.wavelength_bin_widths.at[
                            :-1
                        ].set(jnp.diff(self.wavelengths))
                        self.wavelength_bin_widths = self.wavelength_bin_widths.at[
                            -1
                        ].set(self.wavelength_bin_widths[-2])
            else:
                if self.photometric_bin_edges is None:
                    raise ValueError(
                        f"'photometric_bin_edges' must be a tuple of two floats in photometric data ('{self.name}'), "
                        f"but is None"
                    )

                if wavelength_boundaries is not None:
                    self.wavelength_boundaries = wavelength_boundaries
                else:
                    self.wavelength_boundaries = (
                        0.95 * self.photometric_bin_edges[0],
                        1.05 * self.photometric_bin_edges[1],
                    )

                # For binning later
                # TODO @Evert why using a float instead of an array of size 1? The attribute would be more predictable
                self.wavelength_bin_widths = (
                    self.photometric_bin_edges[1] - self.photometric_bin_edges[0]
                )

                if self.data_resolution is None:
                    self.data_resolution = (
                        jnp.mean(self.photometric_bin_edges)
                        / self.wavelength_bin_widths
                    )

            self._validate_covariance_configuration()

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

        if self.photometry:
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
    def loadfits(self, path) -> None:
        """
        Load a Radtrans-formatted fits file.
        Must include extension SPECTRUM with fields WAVELENGTH, FLUX
        and COVARIANCE (or ERROR).

        Args:
            path : str
                Directory and filename of the data.
        """

        if self.photometry:
            return

        data = fits.getdata(path, "SPECTRUM")

        if hasattr(data, "columns") and data.columns is not None:
            column_names = tuple(data.columns.names)
            self.wavelengths = jnp.asarray(data.field("WAVELENGTH"), dtype=JAX_DTYPE)
            self.spectrum = jnp.asarray(data.field("FLUX"), dtype=JAX_DTYPE)

            if "COVARIANCE" in column_names:
                self.covariance = jnp.asarray(data.field("COVARIANCE"), dtype=JAX_DTYPE)
                self.inv_cov = jnp.linalg.inv(self.covariance)
                sign, self.log_covariance_determinant = jnp.linalg.slogdet(
                    2.0 * jnp.pi * self.covariance
                )
                self.uncertainties = jnp.sqrt(self.covariance.diagonal())
            elif "ERROR" in column_names:
                # Note that this will only be the uncorrelated error.
                # Dot with the correlation matrix (if available) to get
                # the full error.
                self.uncertainties = jnp.asarray(data.field("ERROR"), dtype=JAX_DTYPE)
                self.covariance = jnp.diag(self.uncertainties**2)
                self.inv_cov = jnp.linalg.inv(self.covariance)
        else:
            hdul = fits.open(path)
            self.wavelengths = jnp.asarray(hdul[1].data["WAVELENGTH"], dtype=JAX_DTYPE)
            self.spectrum = jnp.asarray(hdul[1].data["FLUX"], dtype=JAX_DTYPE)

            if "FLUX_STD" in hdul[1].data.columns.names:
                self.uncertainties = jnp.asarray(
                    hdul[1].data["FLUX_STD"], dtype=JAX_DTYPE
                )
            elif "FLUX_ERROR" in hdul[1].data.columns.names:
                self.uncertainties = jnp.asarray(
                    hdul[1].data["FLUX_ERROR"], dtype=JAX_DTYPE
                )

            if "FLUX_COV" in hdul[1].data.columns.names:
                self.covariance = jnp.asarray(hdul[1].data["FLUX_COV"], dtype=JAX_DTYPE)
            elif "COVARIANCE" in hdul[1].data.columns.names:
                self.covariance = jnp.asarray(
                    hdul[1].data["COVARIANCE"], dtype=JAX_DTYPE
                )

            if self.covariance is not None:
                self.inv_cov = jnp.linalg.inv(self.covariance)

        if self.uncertainties is None and self.covariance is not None:
            self.uncertainties = jnp.sqrt(self.covariance.diagonal())

        sign, self.log_covariance_determinant = jnp.linalg.slogdet(
            2.0 * jnp.pi * self.covariance
        )

    def initialise_data_resolution(self, wavelengths_model: ArrayLike) -> None:
        # TODO [4.0.0] this function should raise an error when the interpolation is not possible
        #  As it is now, it can do nothing, which can be confusing
        if isinstance(self.data_resolution, jnp.ndarray):
            self.data_resolution_array_model = jnp.interp(
                wavelengths_model, self.wavelengths, self.data_resolution
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

    def scale_to_distance(self, new_distance: float) -> float:
        """Update the distance variable in the data class.

        This will rescale the flux to the new distance.

        Args:
            new_distance : float
                The distance to the object in CGS units.
        """

        scale: float = (self.system_distance / new_distance) ** 2
        self.spectrum *= scale

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

        if self.photometry:
            raise NotImplementedError(
                f"Covariance fitting is currently unsupported for photometric dataset '{self.name}'"
            )

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
            wavelength_bin_widths=self.wavelength_bin_widths,
            photometry=self.photometry,
            external_radtrans_reference=self.external_radtrans_reference,
            line_opacity_mode=self.line_opacity_mode,
            model_contract=model_contract,
            scale_flux=self.scale_flux,
            scale_uncertainties=self.scale_uncertainties,
            fit_flux_offset=self.fit_flux_offset,
            fit_instrumental_resolution=self.fit_instrumental_resolution,
            subtract_continuum=self.subtract_continuum,
            photometric_transformation_function=self.photometric_transformation_function,
            fit_covariance=self.fit_covariance,
            covariance_mode=self.covariance_mode,
            global_covariance_kernel=self.global_covariance_kernel,
            local_covariance_kernel=self.local_covariance_kernel,
            n_local_covariance_kernels=self.n_local_covariance_kernels,
            covariance_jitter=self.covariance_jitter,
            variability_atmospheric_column_model_flux_return_mode=(
                self.variability_atmospheric_column_model_flux_return_mode
            ),
            metadata={
                "system_distance": self.system_distance,
                "radtrans_grid": self.radtrans_grid,
                "photometric_bin_edges": self.photometric_bin_edges,
                "wavelength_boundaries": self.wavelength_boundaries,
                "atmospheric_column_flux_mixer": self.atmospheric_column_flux_mixer,
                "concatenate_flux_epochs_variability": self.concatenate_flux_epochs_variability,
            },
        )

    def to_likelihood_state(self, n_free_parameters: int = 0) -> LikelihoodState:
        return LikelihoodState(
            data_uncertainties=self.uncertainties,
            data_covariance=self.covariance,
            mask=self.mask,
            inv_covariance=self.inv_cov,
            log_covariance_determinant=self.log_covariance_determinant,
            n_free_parameters=n_free_parameters,
            metadata={
                "name": self.name,
                "has_covariance": self.covariance is not None,
                "fit_covariance": self.fit_covariance,
                "covariance_mode": self.covariance_mode,
                "photometry": self.photometry,
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
        photometric_bin_edges,
        path_to_observations=None,
        data_resolution=None,
        model_resolution=None,
        system_distance=None,
        external_radtrans_reference=None,
        model_generating_function=None,
        wavelength_boundaries=None,
        scale_flux=False,
        scale_uncertainties=False,
        fit_flux_offset=False,
        line_opacity_mode="c-k",
        radtrans_object=None,
        wavelengths=None,
    ):

        self.photometric_transformation_function = photometric_transformation_function
        self.photometric_bin_edges = jnp.asarray(photometric_bin_edges)

        if self.photometric_transformation_function is None:
            raise ValueError(
                f"{name}: photometric_transformation_function must be provided for PhotometryData."
            )

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

        # Determine wavelength_boundaries for the base RadTrans object
        # If not provided, derive from the full range of photometric bins.
        derived_wavelength_boundaries = wavelength_boundaries
        if derived_wavelength_boundaries is None:
            if self.photometric_bin_edges.ndim == 1:  # Single band
                derived_wavelength_boundaries = [
                    0.95 * self.photometric_bin_edges[0],
                    1.05 * self.photometric_bin_edges[1],
                ]
            else:  # Multiple bands
                derived_wavelength_boundaries = [
                    0.95 * jnp.min(self.photometric_bin_edges[:, 0]),
                    1.05 * jnp.max(self.photometric_bin_edges[:, 1]),
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
            system_distance=system_distance,
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

        if wavelength_boundaries is not None:
            self.wavelength_boundaries = wavelength_boundaries
        else:
            self.wavelength_boundaries = [
                0.95 * self.photometric_bin_edges[0],
                1.05 * self.photometric_bin_edges[1],
            ]
        # For binning later
        self.wavelength_bin_widths = (
            self.photometric_bin_edges[1] - self.photometric_bin_edges[0]
        )
        if self.data_resolution is None:
            self.data_resolution = (
                jnp.mean(self.photometric_bin_edges) / self.wavelength_bin_widths
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
        system_distance=None,
        external_radtrans_reference=None,
        model_generating_function=None,
        wavelength_boundaries=None,
        scale_flux=False,
        scale_uncertainties=False,
        fit_flux_offset=False,
        line_opacity_mode="c-k",
        radtrans_grid=False,  # TODO: get rid of this as a parameter?
        concatenate_flux_epochs_variability=False,
        atmospheric_column_flux_mixer=None,  # Split to TimeSeriesData
        variability_atmospheric_column_model_flux_return_mode=False,  # Split to TimeSeriesData
        radtrans_object=None,
        spectrum=None,  # Split to SpectrumData
        wavelengths=None,  # Split to SpectrumData
        wavelength_bin_widths=None,  # Split to SpectrumData
        uncertainties=None,  # Split to SpectrumData
        covariance=None,
        mask=None,  # Split to SpectrumData
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
            system_distance=system_distance,
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

    def to_observation_state(self, model_contract: str = "legacy") -> ObservationState:
        obs = super().to_observation_state(model_contract=model_contract)
        updated_metadata = dict(obs.metadata)
        updated_metadata["observation_times"] = self.observation_times
        if self.time_series_config is not None:
            updated_metadata["time_series_config"] = self.time_series_config
        return ObservationState(
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
            wavelength_bin_widths=obs.wavelength_bin_widths,
            photometry=obs.photometry,
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
