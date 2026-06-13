"""
Module for downloading filter data from the website
of the SVO Filter Profile Service. Reimplemented from
species: https://github.com/tomasstolker/species to
avoid circular dependencies between species and petitRADTRANS.
"""

import os

import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

from typing import Tuple, Optional, Union
import urllib.request
import requests
import warnings
import pooch
from pathlib import Path
import h5py
from filelock import FileLock

from astropy.io.votable import parse_single_table
from astropy.io import fits

from petitRADTRANS.config.configuration import petitradtrans_config_parser


def _db_lock(database_path: str) -> FileLock:
    return FileLock(database_path + ".lock", timeout=120)


class ReadFilter:
    """
    Class for reading a filter profile from the database.
    """

    def __init__(self, filter_name: str) -> None:
        """
        Parameters
        ----------
        filter_name : str
            Filter name as stored in the database. Filter names from
            the SVO Filter Profile Service will be automatically
            downloaded, stored in the database, and read from the
            database.

        Returns
        -------
        NoneType
            None
        """

        self.filter_name = filter_name

        config = petitradtrans_config_parser

        config.init_filter_database()

        self.database = config["Filters"]["database"]
        self.data_folder = config["Filters"]["data_folder"]
        self.vega_mag = config["Filters"]["vega_mag"]

        with h5py.File(self.database, "r") as hdf5_file:
            # Check if the filter is found in 'r' mode
            # because the 'a' mode is not possible when
            # using multiprocessing
            if f"filters/{self.filter_name}" in hdf5_file:
                filter_found = True
            else:
                filter_found = False

        if not filter_found:
            with _db_lock(self.database):
                with h5py.File(self.database, "r") as hdf5_file:
                    filter_found = f"filters/{self.filter_name}" in hdf5_file
                if not filter_found:
                    with h5py.File(self.database, "a") as hdf5_file:
                        add_filter_profile(self.data_folder, hdf5_file, self.filter_name)

    def get_filter(self) -> np.ndarray:
        """
        Select a filter profile from the database.

        Returns
        -------
        np.ndarray
            Array with the wavelengths and filter transmission. The
            array has 2 dimensions with the shape (n_wavelengths, 2).
        """

        with h5py.File(self.database, "r") as hdf5_file:
            data = np.asarray(hdf5_file[f"filters/{self.filter_name}"])

            if data.shape[0] == 2 and data.shape[1] > data.shape[0]:
                # Required for backward compatibility
                data = np.transpose(data)

        return data

    def interpolate_filter(self) -> interp1d:
        """
        Interpolate a filter profile with the `interp1d <https://
        docs.scipy.org/doc/scipy/reference/generated/
        scipy.interpolate.interp1d.html>`_ function from
        ``scipy.interpolate`` and linear kind of interpolation.

        Returns
        -------
        scipy.interpolate.interp1d
            Linearly interpolated filter profile.
        """

        data = self.get_filter()

        return interp1d(
            data[:, 0],
            data[:, 1],
            kind="linear",
            bounds_error=False,
            fill_value=float("nan"),
        )

    def wavelength_range(
        self,
    ) -> Tuple[Union[np.float32, np.float64], Union[np.float32, np.float64]]:
        """
        Extract the wavelength range of the filter profile.

        Returns
        -------
        float
            Minimum wavelength (:math:`\\mu\\mathrm{m}`).
        float
            Maximum wavelength (:math:`\\mu\\mathrm{m}`).
        """

        data = self.get_filter()

        return data[0, 0], data[-1, 0]

    def mean_wavelength(self) -> Union[np.float32, np.float64]:
        """
        Calculate the weighted mean wavelength of the filter profile.

        Returns
        -------
        float
            Mean wavelength (:math:`\\mu\\mathrm{m}`).
        """

        data = self.get_filter()

        return np.trapezoid(data[:, 0] * data[:, 1], x=data[:, 0]) / np.trapezoid(
            data[:, 1], x=data[:, 0]
        )

    def effective_wavelength(self) -> Union[np.float32, np.float64]:
        """
        Calculate the effective wavelength of the filter profile.
        The effective wavelength is calculated as the weighted
        average based on the filter profile and the spectrum of Vega.

        Returns
        -------
        float
            Effective wavelength (:math:`\\mu\\mathrm{m}`).
        """

        filter_profile = self.get_filter()

        with h5py.File(self.database, "r") as hdf5_file:
            vega_found = "spectra/calibration/vega" in hdf5_file

        if not vega_found:
            with _db_lock(self.database):
                with h5py.File(self.database, "r") as hdf5_file:
                    vega_found = "spectra/calibration/vega" in hdf5_file
                if not vega_found:
                    with h5py.File(self.database, "a") as hdf5_file:
                        add_vega(self.data_folder, hdf5_file)

        with h5py.File(self.database, "r") as hdf5_file:
            vega_spec = np.array(hdf5_file["spectra/calibration/vega"])

        flux_interp = interp1d(
            vega_spec[0,],
            vega_spec[1,],
            bounds_error=False,
            fill_value="extrapolate",
        )

        flux_filter = flux_interp(filter_profile[:, 0])

        return np.trapezoid(
            filter_profile[:, 0] * filter_profile[:, 1] * flux_filter,
            x=filter_profile[:, 0],
        ) / np.trapezoid(filter_profile[:, 1] * flux_filter, x=filter_profile[:, 0])

    def filter_fwhm(self) -> Optional[float]:
        """
        Calculate the full width at half maximum (FWHM)
        of the filter profile.

        Returns
        -------
        float, None
            Full width at half maximum (:math:`\\mu\\mathrm{m}`).
            Returns ``None`` if the filter has only one wavelength.
            with a non-zero transmission.
        """

        data = self.get_filter()

        if len(np.nonzero(data[:, 1])[0]) > 1:
            spline = InterpolatedUnivariateSpline(
                data[:, 0], data[:, 1] - np.max(data[:, 1]) / 2.0
            )
            root = spline.roots()

            diff = root - self.mean_wavelength()

            root1 = np.amax(diff[diff < 0.0])
            root2 = np.amin(diff[diff > 0.0])

            filt_fwhm = root2 - root1

        else:
            filt_fwhm = None

        return filt_fwhm

    def effective_width(self) -> Union[np.float32, np.float64]:
        """
        Calculate the effective width of the filter profile. The
        effective width is equivalent to the horizontal size of a
        rectangle with height equal to the maximum transmission and
        with the same area as the one covered by the filter profile.

        Returns
        -------
        float
            Effective width (:math:`\\mu\\mathrm{m}`).
        """

        data = self.get_filter()

        return np.trapezoid(data[:, 1], x=data[:, 0]) / np.amax(data[:, 1])

    def detector_type(self) -> str:
        """
        Return the detector type.

        Returns
        -------
        str
            Detector type ('energy' or 'photon').
        """

        with h5py.File(self.database, "r") as hdf5_file:
            dset = hdf5_file[f"filters/{self.filter_name}"]

            if "det_type" in dset.attrs:
                det_type = dset.attrs["det_type"]

            else:
                warnings.warn(
                    f"Detector type not found for {self.filter_name}. "
                    "The database was probably created before the "
                    "detector type was introduced in species v0.3.1. "
                    "Assuming a photon-counting detector."
                )

                det_type = "photon"

        return det_type


class SyntheticPhotometry:
    """
    Class for calculating synthetic photometry from a spectrum and also
    for converting between magnitudes and fluxes. Any filter from the
    `SVO Filter Profile Service <http://svo2.cab.inta-csic.es/svo/
    theory/fps/>`_ will be automatically downloaded and added to the
    database. Also the detector type (energy- or photon-counting) will
    be fetched. For a photon-counting detector, an additional
    wavelength factor is included in the integral for calculating the
    synthetic photometry, although typically the impact of the factor
    on the calculated flux is negligible. It is also important to note
    that by default the magnitude of Vega is set to 0.03 for all
    filters. The value can be adjusted in the `configuration file
    <https://species.readthedocs.io/en/latest/configuration.html>`_.
    """

    def __init__(self, filter_name: str, zero_point: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        filter_name : str
            Filter name by which the profile is stored in database.
            Any filter from the `SVO Filter Profile Service
            <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_ will be
            automatically downloaded and added to the database.
        zero_point : float, None
            Zero-point flux (:math:`\\mathrm{W}`
            :math:`\\mathrm{m}^{-2}` :math:`\\mu\\mathrm{m}^{-1}`) for
            ``filter_name``. This flux is equalized to the magnitude of
            Vega, which by default is set to 0.03 for all filters. The
            value can be adjusted in the `configuration file <https://
            species.readthedocs.io/en/latest/configuration.html>`_.
            By default, the argument of ``zero_point`` is set to
            ``None``, in which case the zero point is calculated
            internally. The zero point can be accessed through
            ``zero_point`` attribute from instance of
            :class:`~species.phot.syn_phot.SyntheticPhotometry`.

        Returns
        -------
        NoneType
            None
        """

        self.filter_name = filter_name
        self.zero_point = zero_point
        self.filter_interp = None
        self.wavel_range = None

        config = petitradtrans_config_parser
        config.init_filter_database()

        self.database = config["Filters"]["database"]
        self.data_folder = config["Filters"]["data_folder"]
        self.vega_mag = float(config["Filters"]["vega_mag"])

        read_filt = ReadFilter(self.filter_name)
        self.det_type = read_filt.detector_type()

        if self.zero_point is None:
            self.zero_point = self.calc_zero_point()

        else:
            warnings.warn(
                "Please note that a manually provided zero-point flux "
                "is by default equalized to a magnitude of 0.03 for "
                "all filters. The magnitude of Vega can be adjusted "
                "in the configuration file by "
                "setting the 'vega_mag' parameter. Currently the "
                f"parameter is set to {self.vega_mag}."
            )

    def calc_zero_point(self) -> Union[float, np.float64]:
        """
        Internal function for calculating the zero point of the
        provided ``filter_name``. The zero point is here defined
        as the flux of Vega, which by default is set to a
        magnitude of 0.03 for all filters.

        Returns
        -------
        float
            Zero-point flux (:math:`\\mathrm{W}`
            :math:`\\mathrm{m}^{-2}` :math:`\\mu\\mathrm{m}^{-1}`).
        """

        if self.wavel_range is None:
            read_filt = ReadFilter(self.filter_name)
            self.wavel_range = read_filt.wavelength_range()

        with h5py.File(self.database, "r") as hdf5_file:
            vega_found = "spectra/calibration/vega" in hdf5_file

        if not vega_found:
            with _db_lock(self.database):
                with h5py.File(self.database, "r") as hdf5_file:
                    vega_found = "spectra/calibration/vega" in hdf5_file
                if not vega_found:
                    with h5py.File(self.database, "a") as hdf5_file:
                        add_vega(self.data_folder, hdf5_file)

        with h5py.File(self.database, "r") as hdf5_file:
            vega_spec = np.array(hdf5_file["spectra/calibration/vega"])

        wavelength = vega_spec[0,]
        flux = vega_spec[1,]

        wavelength_crop = wavelength[
            (wavelength > self.wavel_range[0]) & (wavelength < self.wavel_range[1])
        ]

        flux_crop = flux[
            (wavelength > self.wavel_range[0]) & (wavelength < self.wavel_range[1])
        ]

        return self.spectrum_to_flux(wavelength_crop, flux_crop)[0]

    def spectrum_to_flux(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        error: Optional[np.ndarray] = None,
        threshold: Optional[float] = 0.01,
    ) -> Tuple[
        Union[float, np.float32, np.float64],
        Optional[Union[float, np.float32, np.float64]],
    ]:
        """
        Function for calculating the average flux from a spectrum and
        a filter profile. The uncertainty is propagated by sampling
        200 random values from the error distributions.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength points (um).
        flux : np.ndarray
            Flux (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        error : np.ndarray, None
            Uncertainty (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`). Not used if set to ``None``.
        threshold : float, None
            Transmission threshold (value between 0 and 1). If the
            minimum transmission value is larger than the threshold,
            a NaN is returned. This will happen if the input spectrum
            does not cover the full wavelength range of the filter
            profile. The parameter is not used if set to ``None``
            (default: 0.01).

        Returns
        -------
        float
            Average flux (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        float, None
            Uncertainty (:math:`\\mathrm{W}` :math:`\\mathrm{m}^{-2}`
            :math:`\\mu\\mathrm{m}^{-1}`).
        """

        # Remove fluxes that are a NaN

        nan_idx = np.isnan(flux)

        if np.sum(nan_idx) > 0:
            warnings.warn(
                f"Found {np.sum(nan_idx)} fluxes with NaN. Removing "
                "these spectral fluxes from the input data before "
                "calculating synthetic photometry."
            )

            wavelength = wavelength[~nan_idx]
            flux = flux[~nan_idx]

            if error is not None:
                error = error[~nan_idx]

        if error is not None:
            # The error calculation requires the original
            # spectrum because spectrum_to_flux is used
            wavel_error = wavelength.copy()
            flux_error = flux.copy()

        if self.filter_interp is None:
            # Interpolate filter profile
            read_filt = ReadFilter(self.filter_name)
            self.filter_interp = read_filt.interpolate_filter()

            if self.wavel_range is None:
                # Set the wavel_range attribute to the
                # wavelength range of the filter profile
                self.wavel_range = read_filt.wavelength_range()

        if wavelength.size == 0:
            syn_flux = np.nan

            if error is not None:
                error_flux = np.nan
            else:
                error_flux = None

            indices = None

            warnings.warn(
                f"Calculation of the mean flux for {self.filter_name} "
                "is not possible because the wavelength array is "
                "empty. Returning a NaN for the flux."
            )

        else:
            indices = np.where(
                (self.wavel_range[0] <= wavelength)
                & (wavelength <= self.wavel_range[1])
            )[0]

        if indices is not None and indices.size < 2:
            syn_flux = np.nan

            if error is not None:
                error_flux = np.nan
            else:
                error_flux = None

            warnings.warn(
                "Calculating a synthetic flux requires more than "
                "one wavelength point. Photometry is set to NaN."
            )

        else:
            if threshold is None and (
                wavelength[0] > self.wavel_range[0]
                or wavelength[-1] < self.wavel_range[1]
            ):
                warnings.warn(
                    f"The filter profile of {self.filter_name} "
                    f"({self.wavel_range[0]:.4f}-"
                    f"{self.wavel_range[1]:.4f}) extends beyond "
                    f"the wavelength range of the spectrum "
                    f"({wavelength[0]:.4f}-{wavelength[-1]:.4f}). "
                    "The synthetic flux is set to NaN. Setting "
                    "the 'threshold' parameter will loosen the "
                    "wavelength constraints."
                )

                syn_flux = np.nan

            else:
                wavelength = wavelength[indices]
                flux = flux[indices]

                transmission = self.filter_interp(wavelength)

                if np.sum(wavelength) == 0.0:
                    # The wavelength array looks empty but it is and
                    # empty array inside another array so the size
                    # of the wavelength array is 1. The sum however
                    # is 0 so that is used to check if it is empty
                    warnings.warn(
                        f"The filter profile of {self.filter_name} "
                        f"({self.wavel_range[0]:.4f}-{self.wavel_range[1]:.4f}) "
                        f"lies outside the wavelength range of the spectrum. "
                        f"The returned synthetic flux is therefore set to NaN."
                    )

                    syn_flux = np.nan

                elif (
                    threshold is not None
                    and (transmission[0] > threshold or transmission[-1] > threshold)
                    and (
                        wavelength[0] < self.wavel_range[0]
                        or wavelength[-1] > self.wavel_range[-1]
                    )
                ):
                    warnings.warn(
                        f"The filter profile of {self.filter_name} "
                        f"({self.wavel_range[0]:.4f}-{self.wavel_range[1]:.4f}) "
                        f"extends beyond the wavelength range of the spectrum "
                        f"({wavelength[0]:.4f}-{wavelength[-1]:.4f}). The flux "
                        f"is set to NaN. Increasing the 'threshold' parameter "
                        f"({threshold}) will loosen the wavelength constraint."
                    )

                    syn_flux = np.nan

                else:
                    indices = np.isnan(transmission)
                    indices = np.logical_not(indices)

                    if self.det_type == "energy":
                        # Energy counting detector
                        integrand1 = transmission[indices] * flux[indices]
                        integrand2 = transmission[indices]

                    elif self.det_type == "photon":
                        # Photon counting detector
                        integrand1 = (
                            wavelength[indices] * transmission[indices] * flux[indices]
                        )
                        integrand2 = wavelength[indices] * transmission[indices]

                    integral1 = np.trapezoid(integrand1, x=wavelength[indices])
                    integral2 = np.trapezoid(integrand2, x=wavelength[indices])

                    syn_flux = integral1 / integral2

        if error is not None and not np.any(np.isnan(error)) and not np.isnan(syn_flux):
            phot_random = np.zeros(200)

            for i in range(200):
                # Use the original spectrum size (i.e. wavel_error and flux_error)
                spec_random = (
                    flux_error
                    + np.random.normal(loc=0.0, scale=1.0, size=wavel_error.shape[0])
                    * error
                )

                phot_random[i] = self.spectrum_to_flux(
                    wavel_error, spec_random, error=None, threshold=threshold
                )[0]

            nan_idx = np.isnan(phot_random)

            if np.sum(nan_idx) > 0:
                warnings.warn(
                    f"{np.sum(nan_idx)} out of 200 samples "
                    "that are used for estimating the "
                    "uncertainty on the synthetic flux "
                    "are NaN so removing these samples."
                )

                phot_random = phot_random[~nan_idx]

            error_flux = np.std(phot_random)

        elif error is not None and np.any(np.isnan(error)):
            warnings.warn("Spectum contains NaN so cannot calculate the error.")
            error_flux = None

        else:
            error_flux = None

        return syn_flux, error_flux


def add_vega(input_path: str, database: h5py._hl.files.File) -> None:
    """
    Function for adding a flux-calibrated spectrum of Vega to the
    database. The latest spectrum (alpha_lyr_stis_011.fits) is
    downloaded from the STScI archive (see `CALSPEC page <https://
    www.stsci.edu/hst/instrumentation/reference-data-for-calibration
    -and-tools/astronomical-catalogs/calspec>`_ for details).

    Parameters
    ----------
    input_path : str
        Path of the data folder.
    database : h5py._hl.files.File
        Database.

    Returns
    -------
    NoneType
        None
    """

    data_file = Path(input_path) / "alpha_lyr_stis_011.fits"

    if not data_file.exists():
        print()

        try:
            url = (
                "https://archive.stsci.edu/hlsps/reference-atlases"
                "/cdbs/current_calspec/alpha_lyr_stis_011.fits"
            )

            pooch.retrieve(
                url=url,
                known_hash="60aebf5c193223f69061cd176d6309730c3210051fffad0dd6ad44475199ceaa",
                fname="alpha_lyr_stis_011.fits",
                path=input_path,
                progressbar=True,
            )

        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError):
            url = (
                "https://home.strw.leidenuniv.nl/~stolker/"
                "species/alpha_lyr_stis_011.fits"
            )

            pooch.retrieve(
                url=url,
                known_hash="60aebf5c193223f69061cd176d6309730c3210051fffad0dd6ad44475199ceaa",
                fname="alpha_lyr_stis_011.fits",
                path=input_path,
                progressbar=True,
            )

    if "spectra/calibration/vega" in database:
        del database["spectra/calibration/vega"]

    vega_data = fits.getdata(data_file, ext=1)
    wavelength = vega_data["WAVELENGTH"]  # (Angstrom)
    flux = vega_data["FLUX"]  # (erg s-1 cm-2 A-1)
    error_stat = vega_data["STATERROR"]  # (erg s-1 cm-2 A-1)
    error_sys = vega_data["SYSERROR"]  # (erg s-1 cm-2 A-1)

    wavelength *= 1e-4  # (Angstrom) -> (um)
    flux *= 1e-3 * 1e4  # (erg s-1 cm-2 A-1) -> (W m-2 um-1)
    error_stat *= 1e-3 * 1e4  # (erg s-1 cm-2 A-1) -> (W m-2 um-1)
    error_sys *= 1e-3 * 1e4  # (erg s-1 cm-2 A-1) -> (W m-2 um-1)
    error = np.sqrt(error_stat**2 + error_sys**2)  # (W m-2 um-1)

    print("\nAdding spectrum: Vega", end="", flush=True)

    database.create_dataset(
        "spectra/calibration/vega", data=np.vstack((wavelength, flux, error))
    )

    print("\nReference: Bohlin et al. 2014, PASP, 126")
    print("URL: https://ui.adsabs.harvard.edu/abs/2014PASP..126..711B/abstract")


def add_filter_profile(
    input_path: str, database: h5py._hl.files.File, filter_name: str
) -> None:
    """
    Function for downloading and adding a filter profile
    to the HDF5 database.

    Parameters
    ----------
    input_path : str
        Folder where the data is located.
    database : h5py._hl.files.File
        Database.
    filter_name : str
        Filter name from the SVO Filter Profile Service (e.g.,
        'Paranal/NACO.Lp') or a user-defined name if a ``filename``
        is specified.

    Returns
    -------
    None
        NoneType
    """

    wavelength, transmission, detector_type = download_filter(filter_name, input_path)

    if wavelength is not None and transmission is not None:
        wavel_new = [wavelength[0]]
        transm_new = [transmission[0]]

        for i in range(wavelength.size - 1):
            if wavelength[i + 1] > wavel_new[-1]:
                # Required for the issue with the Keck/NIRC2.J filter on SVO
                wavel_new.append(wavelength[i + 1])
                transm_new.append(transmission[i + 1])

        dset = database.create_dataset(
            f"filters/{filter_name}", data=np.column_stack((wavel_new, transm_new))
        )

        dset.attrs["det_type"] = str(detector_type)


def download_filter(
    filter_id: str,
    input_path: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Function for downloading filter profile data
    from the SVO Filter Profile Service.

    Parameters
    ----------
    filter_id : str
        Filter name as listed on the website of the `SVO
        Filter Profile Service
        <http://svo2.cab.inta-csic.es/svo/theory/fps/>`_.
    input_path : str
        Folder where the data is located.

    Returns
    -------
    np.ndarray
        Wavelength (um).
    np.ndarray
        Fractional transmission.
    str
        Detector type ('energy' or 'photon').
    """

    if filter_id == "Magellan/VisAO.rp":
        url = "https://xwcl.science/magao/visao/VisAO_rp_filter_curve.dat"
        filter_path = os.path.join(input_path, "VisAO_rp_filter_curve.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission, _, _ = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "Magellan/VisAO.ip":
        url = "https://xwcl.science/magao/visao/VisAO_ip_filter_curve.dat"
        filter_path = os.path.join(input_path, "VisAO_ip_filter_curve.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission, _, _ = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "Magellan/VisAO.zp":
        url = "https://xwcl.science/magao/visao/VisAO_zp_filter_curve.dat"
        filter_path = os.path.join(input_path, "VisAO_zp_filter_curve.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission, _, _ = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "Keck/NIRC2.NB_4.05":
        # The filter profile of Br_alpha has been digitized from
        # https://www2.keck.hawaii.edu/inst/nirc2/filters.html

        url = "https://home.strw.leidenuniv.nl/~stolker/species/Keck_NIRC2.NB_4.05.dat"
        filter_path = os.path.join(input_path, "Keck_NIRC2.NB_4.05.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "Keck/NIRC.Y":
        # The filter profile of the Y band has been
        # adopted from Hillenbrand et al. (2002)

        url = "https://home.strw.leidenuniv.nl/~stolker/species/Keck_NIRC.Y.dat"
        filter_path = os.path.join(input_path, "Keck_NIRC.Y.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id in ["LCO/VisAO.Ys", "Magellan/VisAO.Ys"]:
        url = "https://xwcl.science/magao/visao/VisAO_Ys_filter_curve.dat"
        filter_path = os.path.join(input_path, "VisAO_Ys_filter_curve.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission, _, _ = np.loadtxt(filter_path, unpack=True)

        # Remove wavelengths with zero transmission
        wavelength = wavelength[:-7]
        transmission = transmission[:-7]

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ELT/METIS.Lp":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/TC_filter_Lp.dat"
        filter_path = os.path.join(input_path, "TC_filter_Lp.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ELT/METIS.Mp":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/TC_filter_Mp.dat"
        filter_path = os.path.join(input_path, "TC_filter_Mp.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ELT/METIS.N1":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/TC_filter_N1.dat"
        filter_path = os.path.join(input_path, "TC_filter_N1.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ELT/METIS.Q1":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/TC_filter_Q1.dat"
        filter_path = os.path.join(input_path, "TC_filter_Q1.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ALMA/band3":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/alma_band3.dat"
        filter_path = os.path.join(input_path, "alma_band3.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ALMA/band6":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/alma_band6.dat"
        filter_path = os.path.join(input_path, "alma_band6.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    elif filter_id == "ALMA/band7":
        url = "https://home.strw.leidenuniv.nl/~stolker/species/alma_band7.dat"
        filter_path = os.path.join(input_path, "alma_band7.dat")
        urllib.request.urlretrieve(url, filter_path)

        wavelength, transmission = np.loadtxt(filter_path, unpack=True)

        det_type = "photon"

        os.remove(filter_path)

    else:
        url = "http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID=" + filter_id
        filter_path = os.path.join(input_path, "filter.xml")
        urllib.request.urlretrieve(url, filter_path)

        try:
            table = parse_single_table(filter_path)

            wavelength = table.array["Wavelength"]
            transmission = table.array["Transmission"]

        except IndexError:
            wavelength = None
            transmission = None
            det_type = None

            warnings.warn(
                f"Filter '{filter_id}' is not available "
                "on the SVO Filter Profile Service."
            )

        except Exception:
            os.remove(filter_path)

            raise ValueError(
                f"The filter data of '{filter_id}' could not "
                "be downloaded. Perhaps the website of the "
                "SVO Filter Profile Service (http://svo2.cab."
                "inta-csic.es/svo/theory/fps/) is not available?"
            )

        if transmission is not None:
            det_type = table.get_field_by_id("DetectorType").value

            # For backward compatibility
            if not isinstance(det_type, str):
                det_type = det_type.decode("utf-8")

            if int(det_type) == 0:
                det_type = "energy"

            elif int(det_type) == 1:
                det_type = "photon"

            else:
                det_type = "photon"

                warnings.warn(
                    f"Detector type, '{det_type}', not "
                    "recognized. Setting detector "
                    "type to photon-counting detector."
                )

            wavelength *= 1e-4  # (um)

        os.remove(filter_path)

    if wavelength is not None:
        indices = []

        for i in range(transmission.size):
            if i == 0 and transmission[i] == 0.0 and transmission[i + 1] == 0.0:
                indices.append(i)

            elif (
                i == transmission.size - 1
                and transmission[i - 1] == 0.0
                and transmission[i] == 0.0
            ):
                indices.append(i)

            elif (
                transmission[i - 1] == 0.0
                and transmission[i] == 0.0
                and transmission[i + 1] == 0.0
            ):
                indices.append(i)

        wavelength = np.delete(wavelength, indices)
        transmission = np.delete(transmission, indices)

        if np.amin(transmission) < 0.0:
            warnings.warn(
                f"The minimum transmission value of {filter_id} is "
                f"smaller than zero ({np.amin(transmission):.2e}). "
                f"Wavelengths with negative transmission "
                f"values will be removed."
            )

            indices = []

            for i, item in enumerate(transmission):
                if item > 0.0:
                    indices.append(i)

            wavelength = wavelength[indices]
            transmission = transmission[indices]

    return wavelength, transmission, det_type
