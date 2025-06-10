import copy
import logging
import os
import warnings

import jax.numpy as jnp
from jax.typing import ArrayLike
from astropy.io import fits

import petitRADTRANS.physical_constants as cst


class Data:
    r"""
    This class stores the spectral data to be retrieved from a single instrument or observation.

    Each dataset is associated with an instance of petitRadTrans and an atmospheric model.
    The pRT instance can be overwritten, and associated with an existing pRT instance with the
    external_pRT_reference parameter.
    This setup allows for joint or independent retrievals on multiple datasets.
    # TODO complete docstring
    Args:
        name : str
            Identifier for this data set.
        path_to_observations : str
            Path to observations file, including filename. This can be a txt or dat file
            containing the wavelength, flux, transit depth and error, or a fits file
            containing the wavelength, spectrum and covariance matrix.
            Alternatively, the data information can be directly given by the wavelengths, spectrum, uncertainties, and
            mask attributes.
        data_resolution : float or ArrayLike
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
            The distance to the object in cgs units. Defaults to a 10pc normalized distance.
        external_radtrans_reference : object
            An existing RadTrans object. Leave as none unless you're sure of what you're doing.
        model_generating_function : method
            A function, typically defined in run_definition.py that returns the model wavelength and spectrum
            (emission or transmission).
            This is the function that contains the physics of the model, and calls pRT in order to compute the
            spectrum.
        wavelength_boundaries : tuple,list
            Set the wavelength range of the pRT object. Defaults to a range +/-5% greater than that of the data.
            Must at
             least be equal to the range of the data.
        scale : bool
            Turn on or off scaling the data by a constant factor. Set to True if scaling the data during the
            retrieval.
        scale_err:
            # TODO complete docstring
        offset_bool:
            # TODO complete docstring
        wavelength_bin_widths : numpy.ndarray
            Set the wavelength bin width to bin the Radtrans object to the data. Defaults to the data bins.
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
            Covariance of the data, in the same units as the spectrum^2.
        mask:
            Mask of the data.
    """
    resolving_power_str = ".R"

    def __init__(self,
                 name,
                 path_to_observations=None,
                 data_resolution=None,
                 model_resolution=None,
                 system_distance=None,
                 external_radtrans_reference=None,
                 model_generating_function=None,
                 wavelength_boundaries=None,
                 scale=False,
                 scale_err=False,
                 offset_bool=False,
                 line_opacity_mode='c-k',
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
                 mask=None  # Split to SpectrumData
                 ):
        self.name = name
        self.path_to_observations = path_to_observations

        # To be filled later
        self.radtrans_object = radtrans_object
        self.wavelengths = wavelengths  #: The wavelength bin centers
        self.spectrum = spectrum  #: The flux or transit depth
        self.uncertainties = uncertainties  #: The error on the flux or transit depth
        self.covariance = covariance
        
        # Add a mask with that will be used in retrievals
        if mask is None:
            self.mask = jnp.zeros(jnp.shape(self.spectrum), dtype=bool)
        else:
            self.mask = mask

        # Sanity check distance
        self.system_distance = system_distance

        if not system_distance:
            self.system_distance = 10. * cst.pc

        if self.system_distance < 1.0 * cst.pc:
            logging.warning("Your system distance is less than 1 pc, are you sure you're using cgs units?")

        self.data_resolution = data_resolution
        self.data_resolution_array_model = None
        self.model_resolution = model_resolution
        self.external_radtrans_reference = external_radtrans_reference
        self.model_generating_function = model_generating_function
        self.line_opacity_mode = line_opacity_mode

        if line_opacity_mode not in ['c-k', 'lbl']:
            raise ValueError(f"line_opacity_mode must be either 'c-k' or 'lbl', but was '{line_opacity_mode}'")

        # Sanity check model function
        if model_generating_function is None and external_radtrans_reference is None:
            raise ValueError(
                f"data '{name}': either model_generating_function or external_radtrans_reference must be set, "
                f"but both were None"
            )

        if model_resolution is not None:
            if line_opacity_mode == 'c_k' and model_resolution > 1000:
                logging.warning("The maximum opacity for c-k mode is 1000!")
                self.model_resolution = None
            if line_opacity_mode == 'lbl' and model_resolution < 1000:
                logging.warning("Your resolution is lower than R=1000, it's recommended to use 'c-k' mode.")

        # Optional, covariance
        self.inv_cov = None
        self.log_covariance_determinant = None

        # Optional, flux and uncertainty scaling
        self.scale = scale
        self.scale_err = scale_err
        self.offset_bool = offset_bool
        self.subtract_continuum = subtract_continuum
        self.scale_factor = 1.0
        self.offset = 0.0
        self.bval = -jnp.inf

        # Bins and photometry
        self.wavelength_boundaries = None
        self.wavelength_bin_widths = wavelength_bin_widths

        self.photometry_range = wavelength_boundaries

        self.radtrans_grid = radtrans_grid
        self.concatenate_flux_epochs_variability = concatenate_flux_epochs_variability
        self.variability_atmospheric_column_model_flux_return_mode = (
            variability_atmospheric_column_model_flux_return_mode)
        self.atmospheric_column_flux_mixer = atmospheric_column_flux_mixer

        # Read in data
        if path_to_observations is not None:
            # Check if data exists
            if not os.path.exists(path_to_observations):
                raise FileNotFoundError(f"data file '{path_to_observations}' does not exist")

            if path_to_observations.endswith("_x1d.fits"):
                self.load_jwst(path_to_observations)
            elif path_to_observations.endswith('.fits'):
                self.loadfits(path_to_observations)
            else:
                self.loadtxt(path_to_observations)

            if wavelength_boundaries is not None:
                self.wavelength_boundaries = wavelength_boundaries
            else:
                self.wavelength_boundaries = [0.95 * self.wavelengths[0],
                                                1.05 * self.wavelengths[-1]]

            if self.wavelength_bin_widths is None:
                if wavelength_bin_widths is not None:
                    self.wavelength_bin_widths = wavelength_bin_widths
                else:
                    self.wavelength_bin_widths = jnp.zeros_like(self.wavelengths)
                    self.wavelength_bin_widths[:-1] = jnp.diff(self.wavelengths)
                    self.wavelength_bin_widths[-1] = self.wavelength_bin_widths[-2]
            else:
                if wavelength_boundaries is not None:
                    self.wavelength_boundaries = wavelength_boundaries
                else:
                    self.wavelength_boundaries = [0.95 * self.photometric_bin_edges[0],
                                                  1.05 * self.photometric_bin_edges[1]]
                # For binning later
                self.wavelength_bin_widths = self.photometric_bin_edges[1] - self.photometric_bin_edges[0]
                if self.data_resolution is None:
                    self.data_resolution = jnp.mean(self.photometric_bin_edges) / self.wavelength_bin_widths

    def loadtxt(self, path, delimiter=',', comments='#'):
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

        if self.photometry:
            return
        obs = jnp.genfromtxt(path, delimiter=delimiter, comments=comments)
        # Input sanity checks
        if jnp.isnan(obs).any():
            obs = jnp.genfromtxt(path)
        if len(obs.shape) < 2:
            obs = jnp.genfromtxt(path, comments=comments)
        if obs.shape[1] == 4:
            self.wavelengths = obs[:, 0]
            self.wavelength_bin_widths = obs[:, 1]
            self.spectrum = obs[:, 2]
            self.uncertainties = obs[:, 3]
            return
        elif obs.shape[1] != 3:
            obs = jnp.genfromtxt(path)

        # Warnings and errors
        if obs.shape[1] < 3:
            raise ValueError(f"data file '{path}' must contain at least 3 columns (wavelength, flux, flux error), "
                             f"but has {obs.shape[1]}")
        elif obs.shape[1] > 4:
            warnings.warn(f"data file '{path}' should contain at most 4 columns "
                          f"(wavelength, [opt, wavelength bins], flux, flux error), "
                          f"but has {obs.shape[1]}\n"
                          f"Additional columns will be ignored")

        if jnp.isnan(obs).any():
            warnings.warn(f"NANs present in data file '{path}'")

        self.wavelengths = obs[:, 0]
        self.spectrum = obs[:, 1]
        self.uncertainties = obs[:, 2]

    def load_x1d_fits(self, path):
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
        self.spectrum = 1e-26 * c_micron * self.spectrum / self.wavelengths ** 2
        self.uncertainties = 1e-26 * c_micron * self.uncertainties / self.wavelengths ** 2


    def loadfits(self, path):
        """
        Load in a particular style of fits file.
        Must include extension SPECTRUM with fields WAVELENGTH, FLUX
        and COVARIANCE (or ERROR).

        Args:
            path : str
                Directory and filename of the data.
        """

        from astropy.io import fits

        if self.photometry:
            return

        self.wavelengths = fits.getdata(path, 'SPECTRUM').field("WAVELENGTH")
        self.spectrum = fits.getdata(path, 'SPECTRUM').field("FLUX")

        try:
            self.covariance = fits.getdata(path, 'SPECTRUM').field("COVARIANCE")
            self.inv_cov = jnp.linalg.inv(self.covariance)
            sign, self.log_covariance_determinant = jnp.linalg.slogdet(2.0 * jnp.pi * self.covariance)
            # Note that this will only be the uncorrelated error.
            # Dot with the correlation matrix (if available) to get
            # the full error.
            try:
                self.uncertainties = fits.getdata(path, 'SPECTRUM').field("ERROR")
            except Exception:  # TODO find what is the error expected here
                self.uncertainties = jnp.sqrt(self.covariance.diagonal())
        except Exception:  # TODO find what is the error expected here
            self.uncertainties = fits.getdata(path, 'SPECTRUM').field("ERROR")
            self.covariance = jnp.diag(self.uncertainties ** 2)
            self.inv_cov = jnp.linalg.inv(self.covariance)

            sign, self.log_covariance_determinant = jnp.linalg.slogdet(2.0 * jnp.pi * self.covariance)


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
        b_val = -jnp.inf
        if parameters is not None:
            if f'{self.name}_b' in parameters.keys():
                b_val = parameters[f'{self.name}_b'].value
            elif 'uncertainty_scaling_b' in parameters.keys():
                b_val = parameters['uncertainty_scaling_b'].value
        return b_val
    
    def set_distance(self, distance):
        """
        Sets the distance variable in the data class.
        This does NOT rescale the flux to the new distance.
        In order to rescale the flux and error, use the scale_to_distance method.

        Args:
            distance : float
                The distance to the object in cgs units.
        """

        self.system_distance = distance
        return self.system_distance

    def initialise_data_resolution(self, wavelengths_model):
        if isinstance(self.data_resolution, ArrayLike):
            self.data_resolution_array_model = jnp.interp(wavelengths_model, self.wavelengths, self.data_resolution)

    def update_bins(self, wlens):
        self.wavelength_bin_widths = jnp.zeros_like(wlens)
        self.wavelength_bin_widths[:-1] = jnp.diff(wlens)
        self.wavelength_bin_widths[-1] = self.wavelength_bin_widths[-2]

    def scale_to_distance(self, new_dist):
        """
        Updates the distance variable in the data class.
        This will rescale the flux to the new distance.

        Args:
            new_dist : float
                The distance to the object in cgs units.
        """

        scale = (self.system_distance / new_dist) ** 2
        self.spectrum *= scale
        if self.covariance is not None:
            self.covariance *= scale ** 2
            self.inv_cov = jnp.linalg.inv(self.covariance)
            sign, self.log_covariance_determinant = jnp.linalg.slogdet(2.0 * jnp.pi * self.covariance)

            self.uncertainties = jnp.sqrt(self.covariance.diagonal())
        else:
            self.uncertainties *= scale
            self.covariance = jnp.diag(self.uncertainties)
            self.inv_cov = jnp.linalg.inv(self.covariance)
            sign, self.log_covariance_determinant = jnp.linalg.slogdet(2.0 * jnp.pi * self.covariance)
        self.system_distance = new_dist
        return scale

    def scale_and_offset_flux(self, parameters_dict: dict) -> ArrayLike:
        """
        Applies scaling and offset to self.spectrum based on parameters in parameters_dict.
        """
        scaled_spectrum = self.spectrum.copy()  # Operate on a copy

        scale_factor = 1.0
        if self.scale:
            param_val = parameters_dict.get(f"{self.name}_scale_factor")
            if param_val is not None:
                scale_factor = param_val.value if hasattr(param_val, 'value') else param_val

        scaled_spectrum = scaled_spectrum * scale_factor

        offset_val = 0.0
        if self.offset_bool:
            param_val = parameters_dict.get(f"{self.name}_offset")
            if param_val is not None:
                offset_val = param_val.value if hasattr(param_val, 'value') else param_val

        scaled_spectrum = scaled_spectrum - offset_val
        return scaled_spectrum

    def scale_and_inflate_uncertainties(self, parameters_dict: dict) -> tuple[ArrayLike, ArrayLike]:
        """
        Applies scaling to self.uncertainties or self.covariance.
        Returns a tuple: (scaled_uncertainties, scaled_covariance). One will be None.
        """
        scaled_uncertainties = None
        scaled_covariance = None

        # Multiplicative scaling factor for error bars
        err_scale_factor = 1.0
        if self.scale_err:  # This flag means apply multiplicative scaling to errors
            param_val = parameters_dict.get(f"{self.name}_scale_factor")  # Uses the same param as flux scaling
            if param_val is not None:
                err_scale_factor = param_val.value if hasattr(param_val, 'value') else param_val

        # Additive error inflation term (log10(error_val_to_add_in_variance))
        raw_b_val = self.line_b_uncertainty_scaling(parameters_dict)

        additive_variance_term = 0.0
        if raw_b_val != -jnp.inf:
            additive_variance_term = 10**raw_b_val

        if self.covariance is not None:
            scaled_covariance = self.covariance.copy()
            scaled_covariance = scaled_covariance * (err_scale_factor**2)
            if additive_variance_term > 0:
                # Add to diagonal (variances)
                jnp.fill_diagonal(scaled_covariance, jnp.diag(scaled_covariance) + additive_variance_term)
        elif self.uncertainties is not None:
            scaled_uncertainties = self.uncertainties.copy()
            scaled_uncertainties = scaled_uncertainties * err_scale_factor
            if additive_variance_term > 0:
                scaled_uncertainties = jnp.sqrt(scaled_uncertainties**2 + additive_variance_term)
        else:
            # Should not happen if __init__ ensures either uncertainties or covariance exists
            raise ValueError("Data object has neither uncertainties nor covariance.")

        return scaled_uncertainties, scaled_covariance


class PhotometryData(Data):
    """
    Data class for photometric observations.
    Handles single or multiple photometric bands.
    """
    def __init__(self,
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
                 scale=False,
                 scale_err=False,
                 offset_bool=False,
                 line_opacity_mode='c-k',
                 radtrans_object=None,
                 wavelengths=None):

        self.photometric_transformation_function = photometric_transformation_function
        self.photometric_bin_edges = jnp.asarray(photometric_bin_edges)

        if self.photometric_transformation_function is None:
            raise ValueError(f"{name}: photometric_transformation_function must be provided for PhotometryData.")

        # Handle shape of photometric_bin_edges (single band vs multiple bands)
        if self.photometric_bin_edges.ndim == 1: # Single band [low, high]
            if self.photometric_bin_edges.shape != (2,):
                raise ValueError(f"{name}: photometric_bin_edges for a single band must be a pair [low, high].")
        elif self.photometric_bin_edges.ndim == 2: # Multiple bands [[low1, high1], [low2, high2]]
            if self.photometric_bin_edges.shape[1] != 2:
                raise ValueError(f"{name}: photometric_bin_edges for multiple bands must be an array of pairs [[low, high], ...].")
        else:
            raise ValueError(f"{name}: photometric_bin_edges has invalid shape.")

        # Determine wavelength_boundaries for the base RadTrans object
        # If not provided, derive from the full range of photometric bins.
        derived_wavelength_boundaries = wavelength_boundaries
        if derived_wavelength_boundaries is None:
            if self.photometric_bin_edges.ndim == 1:  # Single band
                derived_wavelength_boundaries = [0.95 * self.photometric_bin_edges[0], 1.05 * self.photometric_bin_edges[1]]
            else: # Multiple bands
                derived_wavelength_boundaries = [0.95 * jnp.min(self.photometric_bin_edges[:, 0]),
                                                 1.05 * jnp.max(self.photometric_bin_edges[:, 1])]
        
        # Calculate effective/central wavelengths and bin widths for base class
        # These will be stored as self.wavelengths and self.wavelength_bin_widths by the base class
        calculated_effective_wavelengths = wavelengths
        calculated_wavelength_bin_widths = None  # To be passed to base

        if self.photometric_bin_edges.ndim == 1: # Single band
            if calculated_effective_wavelengths is None:
                calculated_effective_wavelengths = jnp.mean(self.photometric_bin_edges)
            calculated_wavelength_bin_widths = self.photometric_bin_edges[1] - self.photometric_bin_edges[0]
        else: # Multiple bands
            if calculated_effective_wavelengths is None:
                calculated_effective_wavelengths = jnp.mean(self.photometric_bin_edges, axis=1)
            calculated_wavelength_bin_widths = self.photometric_bin_edges[:, 1] - self.photometric_bin_edges[:, 0]
        
        # Ensure calculated_effective_wavelengths is a 1D array for consistency
        if calculated_effective_wavelengths is not None:
            calculated_effective_wavelengths = jnp.asarray([calculated_effective_wavelengths]).flatten()
        if calculated_wavelength_bin_widths is not None:
            calculated_wavelength_bin_widths = jnp.asarray([calculated_wavelength_bin_widths]).flatten()

        super().__init__(name=name,
                         path_to_observations=path_to_observations,
                         data_resolution=data_resolution,
                         model_resolution=model_resolution,
                         system_distance=system_distance,
                         external_radtrans_reference=external_radtrans_reference,
                         model_generating_function=model_generating_function,
                         wavelength_boundaries=derived_wavelength_boundaries,
                         scale=scale,
                         scale_err=scale_err,
                         offset_bool=offset_bool,
                         wavelength_bin_widths=calculated_wavelength_bin_widths,
                         line_opacity_mode=line_opacity_mode,
                         radtrans_object=radtrans_object,
                         wavelengths=calculated_effective_wavelengths)
        
        if wavelength_boundaries is not None:
            self.wavelength_boundaries = wavelength_boundaries
        else:
            self.wavelength_boundaries = [0.95 * self.photometric_bin_edges[0],
                                          1.05 * self.photometric_bin_edges[1]]
        # For binning later
        self.wavelength_bin_widths = self.photometric_bin_edges[1] - self.photometric_bin_edges[0]
        if self.data_resolution is None:
            self.data_resolution = jnp.mean(self.photometric_bin_edges) / self.wavelength_bin_widths


class TimeSeriesData(Data):
    """
    Data class for photometric observations.
    Handles single or multiple photometric bands.
    """
    def __init__(self,
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
                 scale=False,
                 scale_err=False,
                 offset_bool=False,
                 line_opacity_mode='c-k',
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
                 mask=None  # Split to SpectrumData
                 ):
        self.observation_times = observation_times
        self.filename_list = filename_list
        self.concatenate_flux_epochs_variability = concatenate_flux_epochs_variability
        self.atmospheric_column_flux_mixer = atmospheric_column_flux_mixer
        self.variability_atmospheric_column_model_flux_return_mode =\
            variability_atmospheric_column_model_flux_return_mode

        super().__init__(
                 name=name,
                 path_to_observations=path_to_observations,
                 data_resolution=data_resolution,
                 model_resolution=model_resolution,
                 system_distance=system_distance,
                 external_radtrans_reference=external_radtrans_reference,
                 model_generating_function=model_generating_function,
                 wavelength_boundaries=wavelength_boundaries,
                 scale=scale,
                 scale_err=scale_err,
                 offset_bool=offset_bool,
                 line_opacity_mode=line_opacity_mode,
                 radtrans_grid=radtrans_grid,
                 radtrans_object=radtrans_object,
                 spectrum=spectrum,
                 wavelengths=wavelengths,
                 wavelength_bin_widths=wavelength_bin_widths,
                 uncertainties=uncertainties,
                 covariance=covariance,
                 mask=mask
                 )
    
    def load_time_series_spectrum_txt(self, delimiter=',', comments='#'):
        wavelengths, spectra, bin_widths, uncertainties = [], [], [], []
        for filename in self.filename_list:
            wavelength, spectrum, bin_width, uncertainty = self.loadtxt(self.path_to_observations + filename, delimiter, comments)
            wavelengths.append(wavelength)
            spectra.append(spectrum)
            bin_widths.append(bin_width)
            uncertainties.append(uncertainty)
        self.wavelengths = jnp.array(wavelengths)
        self.spectrum = jnp.array(spectra)
        self.wavelength_bin_widths = jnp.array(bin_widths)
        self.uncertainties = jnp.array(uncertainties)

    def loadtxt(self, path, delimiter=',', comments='#'):
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
        obs = jnp.genfromtxt(path, delimiter=delimiter, comments=comments)
        # Input sanity checks
        if jnp.isnan(obs).any():
            obs = jnp.genfromtxt(path)
        if len(obs.shape) < 2:
            obs = jnp.genfromtxt(path, comments=comments)
        if obs.shape[1] == 4:
            wavelengths = obs[:, 0]
            wavelength_bin_widths = obs[:, 1]
            spectrum = obs[:, 2]
            uncertainties = obs[:, 3]
            return wavelengths, wavelength_bin_widths, spectrum, uncertainties
        elif obs.shape[1] != 3:
            obs = jnp.genfromtxt(path)

        # Warnings and errors
        if obs.shape[1] < 3:
            raise ValueError(f"data file '{path}' must contain at least 3 columns (wavelength, flux, flux error), "
                             f"but has {obs.shape[1]}")
        elif obs.shape[1] > 4:
            warnings.warn(f"data file '{path}' should contain at most 4 columns "
                          f"(wavelength, [opt, wavelength bins], flux, flux error), "
                          f"but has {obs.shape[1]}\n"
                          f"Additional columns will be ignored")

        if jnp.isnan(obs).any():
            warnings.warn(f"NANs present in data file '{path}'")

        wavelengths = obs[:, 0]
        spectrum = obs[:, 1]
        uncertainties = obs[:, 2]
        wavelength_bin_widths = jnp.zeros_like(wavelengths)
        wavelength_bin_widths[:-1] = jnp.diff(wavelengths)
        wavelength_bin_widths[-1] = wavelength_bin_widths[-2]
        return wavelengths, wavelength_bin_widths, spectrum, uncertainties
         