import copy
import logging
import os
import sys

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from petitRADTRANS.fortran_rebin import fortran_rebin as frebin
import petitRADTRANS.physical_constants as cst


class Data:
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
                 wavelength_bin_widths=None,
                 photometry=False,
                 photometric_transformation_function=None,
                 photometric_bin_edges=None,
                 line_opacity_mode='c-k',
                 radtrans_grid=False,
                 concatenate_flux_epochs_variability=False,
                 atmospheric_column_flux_mixer=None,
                 variability_atmospheric_column_model_flux_return_mode=False,
                 radtrans_object=None,
                 wavelengths=None,
                 spectrum=None,
                 uncertainties=None,
                 mask=None
                 ):
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
            data_resolution : float
                Spectral resolution of the instrument. Optional, allows convolution of model to
                instrumental line width.
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
            line_opacity_mode : str
                Should the retrieval be run using correlated-k opacities (default, 'c-k'),
                or line by line ('lbl') opacities? If 'lbl' is selected, it is HIGHLY
                recommended to set the model_resolution parameter. In general,
                'c-k' mode is recommended for retrievals of everything other than
                high-resolution (R>40000) spectra.
        """
        self.name = name
        self.path_to_observations = path_to_observations

        # To be filled later
        self.radtrans_object = radtrans_object
        self.wavelengths = wavelengths  #: The wavelength bin centers
        self.spectrum = spectrum  #: The flux or transit depth
        self.uncertainties = uncertainties  #: The error on the flux or transit depth

        # Add a mask with that will be used in retrievals
        if mask is None:
            self.mask = np.zeros(np.shape(self.spectrum), dtype=bool)
        else:
            self.mask = mask

        # Sanity check distance
        self.system_distance = system_distance

        if not system_distance:
            self.system_distance = 10. * cst.pc

        if self.system_distance < 1.0 * cst.pc:
            logging.warning("Your system distance is less than 1 pc, are you sure you're using cgs units?")

        self.data_resolution = data_resolution
        self.model_resolution = model_resolution
        self.external_radtrans_reference = external_radtrans_reference
        self.model_generating_function = model_generating_function
        self.line_opacity_mode = line_opacity_mode

        if line_opacity_mode not in ['c-k', 'lbl']:
            logging.error("line_opacity_mode must be either 'c-k' or 'lbl'!")
            sys.exit(10)
        # Sanity check model function
        if not model_generating_function and not external_radtrans_reference:
            logging.error("Please provide a model generating function or external reference for " + name + "!")
            sys.exit(8)

        if model_resolution is not None:
            if line_opacity_mode == 'c_k' and model_resolution > 1000:
                logging.warning("The maximum opacity for c-k mode is 1000!")
                self.model_resolution = None
            if line_opacity_mode == 'lbl' and model_resolution < 1000:
                logging.warning("Your resolution is lower than R=1000, it's recommended to use 'c-k' mode.")

        # Optional, covariance and scaling
        self.covariance = None
        self.inv_cov = None
        self.log_covariance_determinant = None
        self.scale = scale
        self.scale_err = scale_err
        self.offset_bool = offset_bool
        self.scale_factor = 1.0
        self.offset = 0.0
        self.bval = -np.inf

        # Bins and photometry
        self.wavelength_boundaries = None
        self.wavelength_bin_widths = wavelength_bin_widths
        self.photometry = photometry
        self.photometric_transformation_function = \
            photometric_transformation_function

        if photometry:  # TODO sys.exit should be avoided, using raise should be preferred
            if photometric_transformation_function is None:
                logging.error("Please provide a photometry transformation function for " + name + "!")
                sys.exit(9)

            if photometric_bin_edges is None:
                logging.error("You must include the photometric bin size if photometry is True!")
                sys.exit(9)

        self.photometry_range = wavelength_boundaries
        self.photometric_bin_edges = photometric_bin_edges

        self.radtrans_grid = radtrans_grid
        self.concatenate_flux_epochs_variability = concatenate_flux_epochs_variability
        self.variability_atmospheric_column_model_flux_return_mode = (
            variability_atmospheric_column_model_flux_return_mode)
        self.atmospheric_column_flux_mixer = atmospheric_column_flux_mixer

        # Read in data
        if path_to_observations is not None:
            # Check if data exists
            if not os.path.exists(path_to_observations):
                logging.error(path_to_observations + " Does not exist!")
                sys.exit(7)

            if not photometry:
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
                        self.wavelength_bin_widths = np.zeros_like(self.wavelengths)
                        self.wavelength_bin_widths[:-1] = np.diff(self.wavelengths)
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
                    self.data_resolution = np.mean(self.photometric_bin_edges) / self.wavelength_bin_widths

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
        obs = np.genfromtxt(path, delimiter=delimiter, comments=comments)
        # Input sanity checks
        if np.isnan(obs).any():
            obs = np.genfromtxt(path)
        if len(obs.shape) < 2:
            obs = np.genfromtxt(path, comments=comments)
        if obs.shape[1] == 4:
            self.wavelengths = obs[:, 0]
            self.wavelength_bin_widths = obs[:, 1]
            self.spectrum = obs[:, 2]
            self.uncertainties = obs[:, 3]
            return
        elif obs.shape[1] != 3:
            obs = np.genfromtxt(path)

        # Warnings and errors
        if obs.shape[1] < 3:
            logging.error("Failed to properly load data in " + path + "!!!")
            sys.exit(6)
        elif obs.shape[1] > 4:
            logging.warning(
                f" File {path} has more than four columns. Retrieval package assumes that "
                f"the first three have this meaning: wavelength, [opt, wavelength bins], flux, flux error")
        if np.isnan(obs).any():
            logging.warning("nans present in " + path + ", please verify your data before running the retrieval!")
        self.wavelengths = obs[:, 0]
        self.spectrum = obs[:, 1]
        self.uncertainties = obs[:, 2]

    def load_jwst(self, path):
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
        self.spectrum = 1e-26 * 2.99792458e14 * self.spectrum / self.wavelengths ** 2
        self.uncertainties = 1e-26 * 2.99792458e14 * self.uncertainties / self.wavelengths ** 2

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
            self.inv_cov = np.linalg.inv(self.covariance)
            sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)
            # Note that this will only be the uncorrelated error.
            # Dot with the correlation matrix (if available) to get
            # the full error.
            try:
                self.uncertainties = fits.getdata(path, 'SPECTRUM').field("ERROR")
            except Exception:  # TODO find what is the error expected here
                self.uncertainties = np.sqrt(self.covariance.diagonal())
        except Exception:  # TODO find what is the error expected here
            self.uncertainties = fits.getdata(path, 'SPECTRUM').field("ERROR")
            self.covariance = np.diag(self.uncertainties ** 2)
            self.inv_cov = np.linalg.inv(self.covariance)

            sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)

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

    def update_bins(self, wlens):
        self.wavelength_bin_widths = np.zeros_like(wlens)
        self.wavelength_bin_widths[:-1] = np.diff(wlens)
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
            self.inv_cov = np.linalg.inv(self.covariance)
            sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)

            self.uncertainties = np.sqrt(self.covariance.diagonal())
        else:
            self.uncertainties *= scale
            self.covariance = np.diag(self.uncertainties)
            self.inv_cov = np.linalg.inv(self.covariance)
            sign, self.log_covariance_determinant = np.linalg.slogdet(2.0 * np.pi * self.covariance)
        self.system_distance = new_dist
        return scale

    def get_chisq(self, wlen_model,
                  spectrum_model,
                  plotting,
                  parameters=None,
                  per_datapoint=False,
                  atmospheric_model_column_fluxes=None
                  ):
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
                    for i_column_flux in range(np.shape(atmospheric_model_column_fluxes)[0]):
                        model_spectra.append(atmospheric_model_column_fluxes[i_column_flux, :])
                else:
                    model_spectra.append(spectrum_model)

                for spectrum_model in model_spectra:
                    if self.data_resolution is not None:
                        spectrum_model = self.convolve(wlen_model,
                                                       spectrum_model,
                                                       self.data_resolution)

                    # Rebin to model observation
                    if np.size(wlen_model) == np.size(self.wavelengths):
                        if np.all(wlen_model == self.wavelengths):
                            flux_rebinned = copy.deepcopy(spectrum_model)
                            rebin = False
                        else:
                            rebin = True
                    else:
                        rebin = True

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
                    column_rebinned_spectra = np.array(column_rebinned_spectra)
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
            f_err = np.sqrt(f_err ** 2 + 10 ** b_val)

        if self.scale_err:
            f_err = f_err * parameters[self.name + "_scale_factor"].value

        log_l = 0.0
        log_l_per_datapoint = None

        if self.covariance is not None:
            inv_cov = self.inv_cov
            log_covariance_determinant = self.log_covariance_determinant

            if self.scale_err:
                cov = self.scale_factor ** 2 * self.covariance
                inv_cov = np.linalg.inv(cov)
                _, log_covariance_determinant = np.linalg.slogdet(2 * np.pi * cov)

            if b_val is not None:
                cov = np.diag(np.diag(self.covariance) + 10 ** b_val)
                inv_cov = np.linalg.inv(cov)
                _, log_covariance_determinant = np.linalg.slogdet(2 * np.pi * cov)

            log_l += -0.5 * np.dot(diff, np.dot(inv_cov, diff))
            log_l += -0.5 * log_covariance_determinant

            if per_datapoint:
                # Following Buerkner et al. (2020) to handle
                # off-diagonal covariance elements
                g_i = np.dot(inv_cov, diff)
                sigma_bar_ii = np.diag(inv_cov)

                sigma_tilde_i = 1 / sigma_bar_ii
                mu_tilde_i = flux_rebinned - g_i / sigma_bar_ii

                log_l_per_datapoint = (
                        -0.5 * np.log(2 * np.pi * sigma_tilde_i)
                        - 0.5 * (flux_rebinned - mu_tilde_i) ** 2 / sigma_tilde_i
                )
        else:
            log_l += -0.5 * np.sum((diff / f_err) ** 2)
            log_l += -0.5 * np.sum(np.log(2.0 * np.pi * f_err ** 2))

            if per_datapoint:  # TODO is there a point calculating log_l if only log_l_per_datapoint is returned?
                # Only diagonal covariance elements
                log_l_per_datapoint = -0.5 * np.log(2 * np.pi * f_err ** 2) - 0.5 * (diff / f_err) ** 2

        if plotting:
            import matplotlib.pyplot as plt

            if not self.photometry:
                plt.clf()
                plt.plot(self.wavelengths, flux_rebinned)
                plt.errorbar(self.wavelengths,
                             self.spectrum * self.scale_factor,
                             yerr=f_err,
                             fmt='+')
                plt.show()

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

            return - 0.5 * data.size * np.log(chi2 / data.size)
        else:
            # Classical log-likelihood
            if beta_mode == 'multiply':
                uncertainties = uncertainties * beta
                penalty_term = - data.size * np.log(beta)
            elif beta_mode == 'add':
                uncertainties = uncertainties + beta
                penalty_term = - np.sum(np.log(uncertainties))
            else:
                raise ValueError(f"beta mode must be 'multiply'|'add', but was '{beta_mode}'")

            chi2 = data - model
            chi2 /= uncertainties
            chi2 *= chi2
            chi2 = chi2.sum()

            return - 0.5 * chi2 + penalty_term

    # TODO: do we want to pass the whole parameter dict,
    # or just set a class variable for b in the likelihood function?
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
        b_val = -np.inf
        if parameters is not None:
            if f'{self.name}_b' in parameters.keys():
                b_val = parameters[f'{self.name}_b'].value
            elif 'uncertainty_scaling_b' in parameters.keys():
                b_val = parameters['uncertainty_scaling_b'].value
        return b_val

    @staticmethod
    def convolve(input_wavelength,
                 input_flux,
                 instrument_res):
        r"""
        This function convolves a model spectrum to the instrumental wavelength
        using the provided data_resolution
        Args:
            input_wavelength : numpy.ndarray
                The wavelength grid of the model spectrum
            input_flux : numpy.ndarray
                The flux as computed by the model
            instrument_res : float
                :math:`\\lambda/\\Delta \\lambda`, the width of the gaussian kernel to convolve with the model spectrum.

        Returns:
            flux_lsf
                The convolved spectrum.
        """

        # From talking to Ignas: delta lambda of resolution element
        # is FWHM of the LSF's standard deviation, hence:
        sigma_lsf = 1. / instrument_res / (2. * np.sqrt(2. * np.log(2.)))

        # The input spacing of petitRADTRANS is 1e3, but just compute
        # it to be sure, or more versatile in the future.
        # Also, we have a log-spaced grid, so the spacing is constant
        # as a function of wavelength
        spacing = np.mean(2. * np.diff(input_wavelength) / (input_wavelength[1:] + input_wavelength[:-1]))

        # Calculate the sigma to be used in the gauss filter in units
        # of input wavelength bins
        sigma_lsf_gauss_filter = sigma_lsf / spacing

        flux_lsf = gaussian_filter(input_flux,
                                   sigma=sigma_lsf_gauss_filter,
                                   mode='nearest')

        return flux_lsf
