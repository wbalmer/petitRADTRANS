import copy
import logging
import os
import warnings

import jax.numpy as jnp
from jax.typing import ArrayLike

from astropy.io import fits
from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
from petitRADTRANS.math import (convolve,
                                convolve_and_sample_Rvers,
                                filter_spectrum_with_spline,
                                median_filter_spectrum,
                                savgol_filter_spectrum)
from spectres.spectral_resampling_numba import spectres_numba

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
         

#### ------- Delete below here ------- ####
    def get_chisq(self,
                  wlen_model,
                  spectrum_model,
                  plotting,
                  parameters=None,
                  per_datapoint=False,
                  atmospheric_model_column_fluxes=None,
                  generate_mock_data=False):
        """
        Calculate the chi square between the model and the data.

        Args:
            wlen_model : numpy.ndarray
                The wavelengths of the model
            spectrum_model : numpy.ndarray
                The model flux in the same units as the data.
            plotting : bool
                Show test plots.
            parameters :
                # TODO complete docstring
            atmospheric_model_column_fluxes : numpy.ndarray
                The fluxes of individual atmospheric columns in case the retrieval is run in the associated
                column_flux_return mode.
            generate_mock_data : bool
                Generate mock data for input = output tests. This is done in get_chisq because here the actual data
                and the forward models are brought to the same shape (resolution convolved, rebinned, column mixed).
        Returns:
            logL : float
                The log likelihood of the model given the data.
        """
        # TODO merge with SpectralModel: the chi2 calculation function should only calculate the chi2
        # Convolve to data resolution
        flux_rebinned = None

        if not self.photometry:
            if self.radtrans_grid:
                # TODO remove the concatenate_flux_epochs_variability treatment: this should now be handled by atmospheric_column_flux_mixer # noqa E501
                # TODO make atmospheric_column_flux_mixer accessible also to self.radtrans_grid = False data.
                if self.concatenate_flux_epochs_variability:
                    flux_rebinned = spectrum_model
                else:
                    index = (wlen_model >= self.wavelengths[0] * 0.99999999) & \
                            (wlen_model <= self.wavelengths[-1] * 1.00000001)
                    if not self.variability_atmospheric_column_model_flux_return_mode:
                        flux_rebinned = spectrum_model[index]
                    elif self.atmospheric_column_flux_mixer is not None:
                        flux_rebinned = self.atmospheric_column_flux_mixer(atmospheric_model_column_fluxes,
                                                                           parameters,
                                                                           self.name)
                        flux_rebinned = flux_rebinned[index]
            else:
                model_spectra = []
                column_rebinned_spectra = []
                if self.atmospheric_column_flux_mixer is not None:
                    for i_column_flux in range(jnp.shape(atmospheric_model_column_fluxes)[0]):
                        model_spectra.append(atmospheric_model_column_fluxes[i_column_flux, :])
                else:
                    model_spectra.append(spectrum_model)

                for spectrum_model in model_spectra:
                    if self.data_resolution_array_model is not None:
                        spectrum_model = convolve(
                            wlen_model,
                            spectrum_model,
                            self.data_resolution_array_model
                        )
                    elif self.data_resolution is not None:
                        spectrum_model = convolve(
                            wlen_model,
                            spectrum_model,
                            self.data_resolution
                        )

                    # Rebin to model observation
                    if jnp.size(wlen_model) == jnp.size(self.wavelengths):
                        if jnp.all(wlen_model == self.wavelengths):
                            flux_rebinned = copy.deepcopy(spectrum_model)
                            rebin = False
                        else:
                            rebin = True
                    else:
                        rebin = True

                    if self.name + "_radial_velocity" in parameters.keys():
                        # RV in km/s -> multiply by 1e5 to cm/s
                        # wlen_model in micron
                        # cst.c in cm/s
                        radial_velocity = parameters[self.name + "_radial_velocity"].value * 1e5 
                        wlen_model *= wlen_model * jnp.sqrt((1 + radial_velocity/cst.c)/(1- radial_velocity/cst.c)) 

                    elif "system_radial_velocity" in parameters.keys():
                        radial_velocity = parameters["system_radial_velocity"].value * 1e5 
                        wlen_model *= jnp.sqrt((1 + radial_velocity/cst.c)/(1 - radial_velocity/cst.c)) 
                        #wlen_model += wavel_shift
                    if rebin:
                        flux_rebinned = frebin.rebin_spectrum_bin(
                            wlen_model,
                            spectrum_model,
                            self.wavelengths,
                            self.wavelength_bin_widths
                        )

                    if self.atmospheric_column_flux_mixer is None:
                        break
                    else:
                        column_rebinned_spectra.append(flux_rebinned)

                if self.atmospheric_column_flux_mixer is not None:
                    column_rebinned_spectra = jnp.array(column_rebinned_spectra)
                    flux_rebinned = self.atmospheric_column_flux_mixer(column_rebinned_spectra,
                                                                       parameters,
                                                                       self.name)
        else:
            flux_rebinned = \
                self.photometric_transformation_function(wlen_model,
                                                         spectrum_model)
            # species spectrum_to_flux functions return (flux,error)
            if isinstance(flux_rebinned, (tuple, list)):
                flux_rebinned = flux_rebinned[0]

        if self.resample:
            r_slope = parameters[self.name + "_R_slope"].value
            r_intersect = parameters[self.name + "_R_int"].value
            r_array = (self.wavelengths*r_slope)+r_intersect
            # TODO: interpolate r_array and use standard convolve function.
            # TODO: compare different convolution and binning methods
            flux_rebinned = convolve_and_sample_Rvers(self.wavelengths, r_array, self.wavelengths, flux_rebinned)
        
        if self.subtract_continuum:
            nodes = parameters[self.name + "_nodes"].value
            x_nodes = jnp.linspace(self.wavelengths[0], self.wavelengths[-1], nodes)
            flux_rebinned = filter_spectrum_with_spline(self.wavelengths,flux_rebinned,x_nodes=x_nodes)

        if self.scale:
            diff = (flux_rebinned - self.spectrum * parameters[self.name + "_scale_factor"].value) + self.offset
        else:
            diff = (flux_rebinned - self.spectrum) + self.offset

        f_err = self.uncertainties
        b_val = None

        if f"{self.name}_b" in parameters.keys():
            b_val = parameters[self.name + "_b"].value
        elif f"{self.name.rsplit('_', 1)[0]}_b" in parameters.keys():
            b_val = parameters[f"{self.name.rsplit('_', 1)[0]}_b"].value
        elif "uncertainty_scaling_b" in parameters.keys():
            b_val = parameters["uncertainty_scaling_b"].value

        if b_val is not None:
            f_err = jnp.sqrt(f_err ** 2 + 10 ** b_val)

        if self.scale_err:
            f_err = f_err * parameters[self.name + "_scale_factor"].value

        log_l = 0.0
        log_l_per_datapoint = None

        if self.covariance is not None:
            inv_cov = self.inv_cov
            log_covariance_determinant = self.log_covariance_determinant

            if self.scale_err:
                cov = self.scale_factor ** 2 * self.covariance
                inv_cov = jnp.linalg.inv(cov)
                _, log_covariance_determinant = jnp.linalg.slogdet(2 * jnp.pi * cov)

            if b_val is not None:
                cov = jnp.diag(jnp.diag(self.covariance) + 10 ** b_val)
                inv_cov = jnp.linalg.inv(cov)
                _, log_covariance_determinant = jnp.linalg.slogdet(2 * jnp.pi * cov)

            log_l += -0.5 * jnp.dot(diff, jnp.dot(inv_cov, diff))
            log_l += -0.5 * log_covariance_determinant

            if per_datapoint:
                # Following Buerkner et al. (2020) to handle
                # off-diagonal covariance elements
                g_i = jnp.dot(inv_cov, diff)
                sigma_bar_ii = jnp.diag(inv_cov)

                sigma_tilde_i = 1 / sigma_bar_ii
                mu_tilde_i = flux_rebinned - g_i / sigma_bar_ii

                log_l_per_datapoint = (
                        -0.5 * jnp.log(2 * jnp.pi * sigma_tilde_i)
                        - 0.5 * (flux_rebinned - mu_tilde_i) ** 2 / sigma_tilde_i
                )
        else:
            log_l += -0.5 * jnp.sum((diff / f_err) ** 2)
            log_l += -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * f_err ** 2))
            if per_datapoint:  # TODO is there a point calculating log_l if only log_l_per_datapoint is returned?
                # Only diagonal covariance elements
                log_l_per_datapoint = -0.5 * jnp.log(2 * jnp.pi * f_err ** 2) - 0.5 * (diff / f_err) ** 2

        if plotting:
            import matplotlib.pyplot as plt

            if not self.photometry:
                plt.clf()
                plt.title(self.name)
                plt.plot(self.wavelengths, flux_rebinned)
                plt.errorbar(self.wavelengths,
                             self.spectrum * self.scale_factor,
                             yerr=f_err,
                             fmt='+')
                plt.show()

        if generate_mock_data:
            # Check if the mock data folder exists, if not create it:
            if not os.path.exists("mock_data"):
                os.makedirs("mock_data")

            jnp.savetxt("mock_data/" + self.name + "_mock_data.dat",
                       jnp.column_stack((self.wavelengths, flux_rebinned, self.uncertainties)))

        if per_datapoint:
            return log_l_per_datapoint

        return log_l

    def get_log_likelihood(self, spectrum_model):
        """Calculate the log-likelihood between the model and the data.

        The spectrum model must be on the same wavelength grid than the data.

        Args:
            spectrum_model: numpy.ndarray
                The model flux in the same units as the data.

        Returns:
            logL : float
                The log likelihood of the model given the data.
        """
        return self.log_likelihood(
            model=spectrum_model,
            data=self.spectrum,
            uncertainties=self.uncertainties,
            beta=self.scale_factor
        )

    @staticmethod
    def log_likelihood(model, data, uncertainties, beta=None, beta_mode='multiply'):
        """Calculate the log-likelihood between the model and the data.

        The spectrum model must be on the same wavelength grid than the data.

        From Gibson et al. 2020 (https://doi.org/10.1093/mnras/staa228). Constant terms are dropped and constant
        coefficients are set to 1.
        The 'add' beta mode comes from Line et al. 2015 (DOI 10.1088/0004-637X/807/2/183).

        Set:
            chi2(A, B) = sum(((f_i - A * m_i) / (B * sig_i)) ** 2),  implicit sum on i
        with f_i the data, m_i the model, and sig_i the uncertainty; where "i" denotes wavelength/time variation.
        Starting from (implicit product on i):
            L        = prod(1 / sqrt(2 * pi * B * sig_i ** 2) * exp(-1/2 * sum(((f_i - A * m_i) / (B * sig_i)) ** 2))),
            => L     = prod( 1 / sqrt(2 * pi * B * sig_i ** 2) * exp(-1/2 * chi2(A, B)) ),
            => ln(L) = -N/2 * ln(2 * pi) - N * ln(B) - sum(ln(sig_i)) - 1/2 * chi2(A, B).
        Dropping constant terms:
            ln(L)^* = - N * ln(B) - 1/2 * chi2(A, B).

        B can be automatically optimised by nulling the ln(L) partial derivative with respect to B.
        Using the best estimator of B instead of the true value:
            d_ln(L) / d_B = - N / B + 1 / B ** 3 * chi2(A, B=1) = 0,
            => B = sqrt( 1/N *  chi2(A, B=1)).
        Replacing:
            ln(L)^**= - N * ln(B) - 1/2 * chi2(A, B),
                    = - N * ln(sqrt(1/N * chi2(A, B=1))) - 1/2 * sum(((f_i - A * m_i) / (B * sig_i)) ** 2),
                    = - N/2 * ln(1/N * chi2(A, B=1)) - 1/2 / B ** 2 * sum(((f_i - A * m_i) / sig_i) ** 2),  B cst with i
                    = - N/2 * ln(1/N * chi2(A, B=1)) - 1/2 * N / chi2(A, B=1) * chi2(A, B=1),
                    = - N/2 * ln(1/N * chi2(A, B=1)) - N/2.
        Dropping constant terms:
            ln(L)^*** = - N/2 * ln(1/N * chi2(A, B=1)).

        Args:
            model: numpy.ndarray
                The model flux in the same units as the data.
            data: numpy.ndarray
                The data.
            uncertainties: numpy.ndarray
                The uncertainties on the data.
            beta: float, optional
                Noise scaling coefficient. If None, the noise scaling is "automatically optimised".
            beta_mode: string, optional


        Returns:
            logL : float
                The log likelihood of the model given the data.
        """
        if beta is None:
            # "Automatically optimise" for beta
            chi2 = data - model
            chi2 /= uncertainties
            chi2 *= chi2
            chi2 = chi2.sum()

            return - 0.5 * data.size * jnp.log(chi2 / data.size)
        else:
            # Classical log-likelihood
            if beta_mode == 'multiply':
                uncertainties = uncertainties * beta
                penalty_term = - data.size * jnp.log(beta)
            elif beta_mode == 'add':
                uncertainties = uncertainties + beta
                penalty_term = - jnp.sum(jnp.log(uncertainties))
            else:
                raise ValueError(f"beta mode must be 'multiply'|'add', but was '{beta_mode}'")

            chi2 = data - model
            chi2 /= uncertainties
            chi2 *= chi2
            chi2 = chi2.sum()

            return - 0.5 * chi2 + penalty_term

    @staticmethod
    def log_likelihood2chi2(log_likelihood: float) -> float:
        return -2 * log_likelihood

class Rebin:
    """
    A class containing static methods for rebinning and convolving spectra.
    """

    @staticmethod
    def convolve_spectrum(input_wavelength: ArrayLike, input_flux: ArrayLike, instrument_res: any) -> ArrayLike:
        """
        Convolve a model spectrum with a given instrument resolution.
        This method replicates the logic from petitRADTRANS.retrieval.data.Data.convolve().

        Args:
            input_wavelength : ArrayLike
                The wavelength array of the model spectrum.
            input_flux : ArrayLike
                The flux array of the model spectrum.
            instrument_res : float or ArrayLike
                The instrument resolution. Can be a single float (constant resolution)
                or an array (variable resolution per wavelength point).

        Returns:
            flux_lsf : ArrayLike
                The convolved flux array.
        """
        if isinstance(instrument_res, ArrayLike):
            # Variable width convolution
            flux_lsf = fconvolve.variable_width_convolution(
                input_wavelength,
                input_flux,
                instrument_res  # This is expected to be sigma_lsf values per wavelength by variable_width_convolution
            )
        else:
            # Constant resolution convolution using Gaussian filter
            # Calculate sigma_lsf (standard deviation of the LSF)
            sigma_lsf = input_wavelength / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)) * instrument_res)

            # Calculate the spacing of the wavelength array
            spacing = jnp.abs(input_wavelength - jnp.roll(input_wavelength, 1))
            spacing[0] = spacing[1]  # Fix the first element due to jnp.roll

            # Calculate sigma_lsf for gaussian_filter
            sigma_lsf_gauss_filter = sigma_lsf / spacing

            # Convolve with a Gaussian LSF
            flux_lsf = gaussian_filter(input_flux, sigma=sigma_lsf_gauss_filter, mode='nearest')

        return flux_lsf

    @staticmethod
    def rebin_spectrum(
        model_wavelengths: ArrayLike,
        model_spectrum: ArrayLike,
        data_wavelengths: ArrayLike,
        data_wavelength_bin_widths: ArrayLike
    ) -> ArrayLike:
        """
        Rebin a model spectrum to the resolution of the data.
        This method wraps the fortran_rebin.rebin_spectrum_bin function.

        Args:
            model_wavelengths : ArrayLike
                The wavelength array of the model spectrum.
            model_spectrum : ArrayLike
                The flux array of the model spectrum.
            data_wavelengths : ArrayLike
                The wavelength array of the data.
            data_wavelength_bin_widths : ArrayLike
                The widths of the wavelength bins of the data.

        Returns:
            rebinned_flux : ArrayLike
                The rebinned model flux array, on the data_wavelengths grid.
        """
        rebinned_flux = frebin.rebin_spectrum_bin(
            model_wavelengths,
            model_spectrum,
            data_wavelengths,
            data_wavelength_bin_widths
        )
        return rebinned_flux
    
    @staticmethod
    def continuum_subtract_spectrum(
            wavelengths: ArrayLike,
            spectrum: ArrayLike,
            parameters: dict,
            method: str = 'spline',
            name: str = None) -> ArrayLike:
        """
        """

        if method == 'spline':
            nodes = wavelengths[0]
            if name is not None and + (name + "_nodes" in parameters.keys()):
                nodes = parameters[name + "_nodes"].value
            elif "continuum_subtract_nodes" in parameters.keys():
                nodes = parameters["continuum_subtract_nodes"].value
            else:
                raise ValueError("No nodes available to create spline.")
            x_nodes = jjnp.linspace(wavelengths[0], wavelengths[-1], nodes)
            return filter_spectrum_with_spline(wavelengths, spectrum, x_nodes=x_nodes)
        elif method == 'medfilt':
            median_filter_width = 10
            if name is not None and + (name + "_median_filter_width" in parameters.keys()):
                median_filter_width = parameters[name + "_median_filter_width"].value
            elif "median_filter_width" in parameters.keys():
                median_filter_width = parameters["median_filter_width"].value
            return median_filter_spectrum(spectrum, size=median_filter_width)
        elif method == 'savgol':
            window_length = 10
            polyorder = 3
            if name is not None and + (name + "_savgol_window_length" in parameters.keys()):
                window_length = parameters[name + "_savgol_window_length"].value
            elif "savgol_window_length" in parameters.keys():
                window_length = parameters["savgol_window_length"].value

            if name is not None and + (name + "_savgol_polyorder" in parameters.keys()):
                polyorder = parameters[name + "_savgol_polyorder"].value
            elif "savgol_polyorder" in parameters.keys():
                polyorder = parameters["savgol_polyorder"].value
            return savgol_filter_spectrum(spectrum, window_length=window_length, polyorder=polyorder)
