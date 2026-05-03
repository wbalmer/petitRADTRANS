"""
This module re-implements the exo-k correlated-k binning method.
Numba compilation has been replaced with JAX to reduce dependency collisions.
This version only contains the basic functionality necessary
to bin the petitRADTRANS correlated-k tables.

# Exo_k
![Stable Version](https://img.shields.io/pypi/v/exo_k?label=Stable)
![Python Versions](https://img.shields.io/pypi/pyversions/exo_k?label=Python)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Taskfile](https://img.shields.io/badge/Taskfile-grey?logo=task)](https://taskfile.dev/)

Author: Jeremy Leconte (CNRS/LAB/Univ. Bordeaux)

`Exo_k` is a Python 3 based library to handle radiative opacities from various sources for atmospheric applications.

# Acknowledgements

If you use this library in your research, please acknowledge it by citing
[Leconte (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A..20L/abstract):

  * Spectral binning of precomputed correlated-k coefficients. **Astronomy and Astrophysics** 645. Leconte, J. 2021. doi:10.1051/0004-6361/202039040

This project has received funding from the European Research Council (ERC)
under the European Union's Horizon 2020 research and innovation programme
(grant agreement n° 679030/WHIPLASH).

The framework for this documentation has been developped by Aurelien Falco using Sphinx.
The Framework for automatic testing has been developped by Alexandre Mechineau.
"""
import os
from functools import partial

import jax
# Enable 64-bit floats: opacity values can be as small as ~1e-70, which
# underflows silently to zero in JAX's default float32 mode.
jax.config.update("jax_enable_x64", True)
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from datetime import date
import h5py


class Spectral_object(object):
    """A class with some basic functions for all objects with a spectral dimension"""

    def __init__(self):
        """Initializes some attributes to None"""
        self.wns = None
        self.wnedges = None
        self.Nw = None
        self.wn_unit = "cm^-1"

    def select_spectral_range(self, wn_range=None, wl_range=None):
        """Select spectral range, without restricting the data
        (but the spectral axes are modified in place).
        Should use either wn_range OR wl_range, not both.

        To be selected, the whole bin must be inside the range.

        Parameters
        ----------
            wn_range: array, np.ndarray
                Wavenumber range in cm^-1.
            wl_range: array, np.ndarray
                Wavelength range in micron.

        Returns
        -------
            tuple:
                iw_min, iw_max the boundary indices of the spectral range
        """
        if (wn_range is None) and (wl_range is None):
            return None, None
        if wl_range is not None:
            if wn_range is not None:
                raise RuntimeError(
                    "Should provide either wn_range or wl_range, not both!"
                )
            _wn_range = np.sort(10000.0 / np.array(wl_range))
        else:
            _wn_range = np.sort(np.array(wn_range))
        iw_min = np.searchsorted(self.wnedges, _wn_range[0], side="left")
        iw_max = np.searchsorted(self.wnedges, _wn_range[1], side="right")
        iw_max -= 1
        if iw_max <= iw_min:
            raise RuntimeError(f"Spectral range {wn_range} does not contain any point.")
        self.wnedges = self.wnedges[iw_min : iw_max + 1]
        self.wns = self.wns[iw_min:iw_max]
        self.Nw = self.wns.size
        return iw_min, iw_max


class Data_table(Spectral_object):
    """An abstract class that will serve as a basis for Ktable and Xtable.
    This class includes all the interpolation and remapping methods.
    """

    __array_priority__ = 1000
    # this allows __rmul__ to take precedence over numpy array broadcasting in multiplications
    # See https://stackoverflow.com/questions/40694380/
    # forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp/44634634#44634634

    def __init__(self):
        """Initializes all attributes to `None`"""
        super().__init__()
        self.filename = None
        self.mol = None
        self.isotopolog_id = 0
        self.pgrid = None
        self.logpgrid = None
        self.tgrid = None
        self.kdata = None
        self.logk = None
        self.Np = None
        self.Nt = None
        self.Ng = None  # If the table is for xsec, Ng will stay at None.
        # This will allow us to differentiate xsec from corrk
        # when needed in the interpolation routines.
        # Especially to reshape the output.
        self.weights = None
        self.ggrid = None
        self.gedges = None
        self.DOI = "unknown"
        self.sampling_method = "unknown"
        self.Date_ID = (
            "petitRADTRANS" + "0.0.1" + "-" + date.today().strftime("%d/%m/%Y")
        )
        self.Nx = None  # If we are dealing with a Qtable with a variable gas, Nx is the size of the
        # grid along the dimension of the Volume mixing ratio of the variable gas.
        # A None value means that we have either a regular Ktable or a Xtable.
        self.p_unit = "unspecified"
        self.kdata_unit = "unspecified"

    def remove_zeros(self, deltalog_min_value=10.0):
        """Finds zeros in the kdata and set them to (10.**-deltalog_min_value)
        times the minimum positive value in the table (inplace).

        This is to be able to work in logspace.

        Parameters
        ----------
            deltalog_min_value: float
        """
        mask = np.zeros(self.kdata.shape, dtype=bool)
        mask[np.nonzero(self.kdata)] = True
        minvalue = np.amin(self.kdata[mask])
        self.kdata[~mask] = minvalue / (10.0**deltalog_min_value)

    def finalize_init(
        self,
        p_unit="unspecified",
        file_p_unit="unspecified",
        kdata_unit="unspecified",
        file_kdata_unit="unspecified",
        remove_zeros=False,
    ):
        """Common code at the end of the initialization of
        inheriting classes put here to avoid duplicates
        """
        if self.kdata is not None:
            #if self._settings._convert_to_mks:
            #    if p_unit == "unspecified":
            #        p_unit = "Pa"
            #    if kdata_unit == "unspecified":
            #        kdata_unit = "m^2/molecule"
            self.convert_p_unit(p_unit=p_unit, file_p_unit=file_p_unit)
            self.convert_kdata_unit(
                kdata_unit=kdata_unit, file_kdata_unit=file_kdata_unit
            )
            if remove_zeros:
                self.remove_zeros(deltalog_min_value=10.0)

    def convert_p_unit(self, p_unit="unspecified", file_p_unit="unspecified"):
        """Converts pressure to a new unit (inplace).

        Parameters
        ----------
            p_unit: str
                String identifying the pressure units to convert to (e.g. 'bar', 'Pa', 'mbar',
                or any pressure unit recognized by the astropy.units library).
                If ='unspecified', no conversion is done.
            file_p_unit : str, optional
                String to specify the current pressure unit if it is unspecified or if
                you have reasons to believe it is wrong (e.g. you just read a file where
                you know that the pressure grid and the pressure unit do not correspond)
        """
        # if p_unit==file_p_unit and self.p_unit != 'unspecified': return
        current_p_unit = self.p_unit
        self.p_unit, conversion_factor = unit_convert(
            "p_unit", unit_file=current_p_unit, unit_in=file_p_unit, unit_out=p_unit
        )
        self.pgrid = self.pgrid * conversion_factor
        self.logpgrid = np.log10(self.pgrid)

    def convert_kdata_unit(
        self, kdata_unit="unspecified", file_kdata_unit="unspecified"
    ):
        """Converts kdata to a new unit (inplace).

        Parameters
        ----------
            kdata_unit: str
                String to identify the units to convert to. Accepts 'cm^2', 'm^2'
                or any surface unit recognized by the astropy.units library.
                If ='unspecified', no conversion is done.
                In general, kdata should be kept in 'per number' or 'per volume'
                units (as opposed to 'per mass' units) as composition will
                always be assumed to be a number or volume mixing ratio.
                Opacities per unit mass are not supported yet.
                Note that you do not need to specify the '/molec' or
                '/molecule' in the unit.
            file_kdata_unit : str
                String to specify the current kdata unit if it is unspecified or if
                you have reasons to believe it is wrong (e.g. you just read a file where
                you know that the kdata grid and the kdata unit do not correspond)
        """
        # if kdata_unit==file_kdata_unit and self.kdata_unit != 'unspecified': return
        tmp_k_u_in = rm_molec(file_kdata_unit)
        tmp_k_u_out = rm_molec(kdata_unit)
        tmp_k_u_file = rm_molec(self.kdata_unit)
        self.kdata_unit, conversion_factor = unit_convert(
            "kdata_unit",
            unit_file=tmp_k_u_file,
            unit_in=tmp_k_u_in,
            unit_out=tmp_k_u_out,
        )
        self.kdata_unit = self.kdata_unit + "/molecule"
        self.kdata = self.kdata * conversion_factor


class Ktable(Data_table):
    """A class that handles 4D tables of k-coefficients.

    Based on the :class:`~exo_k.data_table.Data_table` class
    that handles basic operations common to Xtable.

    Based on the :class:`~exo_k.ktable_io.Ktable_io` class
    that incorporates all io routines (read/write_xxx for all the xxx supported formats).
    """

    def __init__(
        self,
        filename=None,
        p_unit="unspecified",
        file_p_unit="unspecified",
        kdata_unit="unspecified",
        file_kdata_unit="unspecified",
        remove_zeros=True,
        mol=None,
        **kwargs,
    ):
        """Initializes k coeff table and supporting data from various sources
        (see below by order of precedence)

        Parameters
        ----------
            filename: str, optional
                Relative or absolute name of the file to be loaded.
            filename_filters: sequence of string
                As many strings as necessary to uniquely define
                a file in the global search path defined in
                :class:`~exo_k.settings.Settings`.
                This path will be searched for a file
                with all the filename_filters in the name.
                The filename_filters can contain '*'.
            path: str
                If none of the above is specifed, path can point to
                a directory with a LMDZ type k coeff table.
                In this case, see read_LMDZ for the keywords to specify.

        If none of the parameters above is specified,
        just creates an empty object to be filled later.

        Other Parameters
        ----------------
            p_unit: str, optional
                String identifying the pressure units to convert to (e.g. 'bar', 'Pa', 'mbar',
                or any pressure unit recognized by the astropy.units library).
                If ='unspecified', no conversion is done.
            file_p_unit: str, optional
                String to specify the current pressure unit if it is unspecified or if
                you have reasons to believe it is wrong (e.g. you just read a file where
                you know that the pressure grid and the pressure unit do not correspond)
            kdata_unit: str, optional
                String to identify the unit to convert to.
                Accepts 'cm^2', 'm^2'
                or any surface unit recognized by the astropy.units library.
                If ='unspecified', no conversion is done.
                In general, kdata should be kept in 'per number' or 'per volume'
                units (as opposed to 'per mass' units) as composition will
                always be assumed to be a number or volume mixing ratio.
                Opacities per unit mass are not supported yet.
                Note that you do not need to specify the '/molec' or '/molecule' in the unit.
            file_kdata_unit: str, optional
                String to specify the current kdata unit if it is unspecified or if
                you have reasons to believe it is wrong (e.g. you just read a file where
                you know that the kdata grid and the kdata unit do not correspond)
            remove_zeros: boolean, optional
                If True, the zeros in the kdata table are replaced by
                a value 10 orders of magnitude smaller than the smallest positive value
            mol: str, optional
                The name of the gas or molecule described by the :class:`Ktable`
            search_path: str, optional
                If search_path is provided,
                it locally overrides the global _search_path
                in :class:`~exo_k.settings.Settings`
                and only files in search_path are returned.
        """
        super().__init__()

        self.filename = filename

        if filename is not None:
            self.read_hdf5(filename=filename, mol=mol)

        super().finalize_init(
            p_unit=p_unit,
            file_p_unit=file_p_unit,
            kdata_unit=kdata_unit,
            file_kdata_unit=file_kdata_unit,
            remove_zeros=remove_zeros,
        )

    def read_hdf5(self, filename=None, mol=None, wn_range=None, wl_range=None):
        """Initializes k coeff table and supporting data from an hdf5 file
        (compatible with Exomol format)

        Parameters
        ----------
            file : str
                Name of the input hdf5 file
        """
        if filename is None or not filename.lower().endswith((".hdf5", ".h5")):
            raise RuntimeError("You should provide an input hdf5 file")
        with h5py.File(filename, "r") as f:
            if "mol_name" in f:
                self.mol = f["mol_name"][()]
            elif "mol_name" in f.attrs:
                self.mol = f.attrs["mol_name"]
            else:
                if mol is not None:
                    self.mol = mol
                else:
                    self.mol = os.path.basename(filename).split("_")[0]
            if isinstance(self.mol, np.ndarray):
                self.mol = self.mol[0]
            if isinstance(self.mol, bytes):
                self.mol = self.mol.decode("UTF-8")
            if "method" in f:
                self.sampling_method = f["method"][()][0]
                if isinstance(self.sampling_method, bytes):
                    self.sampling_method = self.sampling_method.decode("UTF-8")
            if "DOI" in f:
                self.DOI = f["DOI"][()][0]
                if isinstance(self.DOI, bytes):
                    self.DOI = self.DOI.decode("UTF-8")
            self.wnedges = f["bin_edges"][...]
            if "bin_centers" in f:
                self.wns = f["bin_centers"][...]
            else:
                self.wns = self.wnedges[:-1] + np.diff(self.wnedges) / 2
            iw_min, iw_max = self.select_spectral_range(wn_range, wl_range)
            if "units" in f["bin_edges"].attrs:
                self.wn_unit = f["bin_edges"].attrs["units"]
            else:
                if "units" in f["bin_centers"].attrs:
                    self.wn_unit = f["bin_centers"].attrs["units"]
            self.kdata = f["kcoeff"][:, :, iw_min:iw_max]
            self.kdata_unit = f["kcoeff"].attrs["units"]
            self.tgrid = f["t"][...]
            self.pgrid = f["p"][...]
            self.logpgrid = np.log10(self.pgrid)
            self.p_unit = f["p"].attrs["units"]
            if "weights" in f.keys():
                self.weights = f["weights"][...]
            else:
                raise RuntimeError("""No weights keyword.
                    This file is probably a cross section file.""")
            self.ggrid = f["samples"][...]
            self.gedges = np.insert(np.cumsum(self.weights), 0, 0.0)
            self.logk = False
        self.Np, self.Nt, self.Nw, self.Ng = self.kdata.shape

    def write_hdf5_common(
        self, f, compression="gzip", compression_level=9, p_unit=None
    ):
        """Method that writes datasets and attributes that are common to
        X and Ktables.

        Parameters
        ----------
            f: h5py file instance
                The file to write that is created in daughter classes
        """
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("DOI", (1,), data=self.DOI, dtype=dt)
        f.create_dataset("Date_ID", (1,), data=self.Date_ID, dtype=dt)
        f.create_dataset("mol_name", (1,), data=self.mol, dtype=dt)
        f.create_dataset(
            "t",
            data=self.tgrid,
            compression=compression,
            compression_opts=compression_level,
        )
        f["t"].attrs["units"] = "K"

        if p_unit is not None:
            conv_factor = u.Unit(self.p_unit).to(u.Unit(p_unit))
            data_to_write = self.pgrid * conv_factor
            f.create_dataset(
                "p",
                data=data_to_write,
                compression=compression,
                compression_opts=compression_level,
            )
            f["p"].attrs["units"] = p_unit
        else:
            f.create_dataset(
                "p",
                data=self.pgrid,
                compression=compression,
                compression_opts=compression_level,
            )
            f["p"].attrs["units"] = self.p_unit

        f.create_dataset(
            "wnrange",
            data=self.wnrange,
            compression=compression,
            compression_opts=compression_level,
        )
        f.create_dataset(
            "wlrange",
            data=self.wlrange,
            compression=compression,
            compression_opts=compression_level,
        )

        for key, val in ktable_long_name_attributes.items():
            if key in f:
                f[key].attrs["long_name"] = val
        for key in ("bin_edges", "bin_centers", "wnrange"):
            if key in f:
                f[key].attrs["units"] = self.wn_unit
        if "wlrange" in f:
            f["wlrange"].attrs["units"] = "micron"

    def write_hdf5(
        self,
        filename,
        compression="gzip",
        compression_level=9,
        kdata_unit=None,
        p_unit=None,
        exomol_units=False,
    ):
        """Saves data in a hdf5 format

        Parameters
        ----------
            filename: str
                Name of the file to be created and saved
            exomol_units: bool (optional)
                If True, data are converted back to
                cm^2 and bar units before being written.
        """
        dt = h5py.string_dtype(encoding="utf-8")
        fullfilename = filename
        if not filename.lower().endswith((".hdf5", ".h5")):
            fullfilename = filename + ".h5"
        folder = os.path.dirname(fullfilename)
        os.makedirs(folder, exist_ok=True)
        with h5py.File(fullfilename, "w") as f:
            f.create_dataset("temperature_grid_type", (1,), data="regular", dtype=dt)
            f.create_dataset("key_iso_ll", (1,), data=self.isotopolog_id)
            f.create_dataset("mol_mass", (1,), data=self.molar_mass * 1000.0)
            f["mol_mass"].attrs["units"] = "AMU"
            if exomol_units:
                kdata_unit = "cm^2/molecule"
                p_unit = "bar"
            if kdata_unit is not None:
                conv_factor = u.Unit(rm_molec(self.kdata_unit)).to(
                    u.Unit(rm_molec(kdata_unit))
                )
                data_to_write = self.kdata * conv_factor
                f.create_dataset(
                    "kcoeff",
                    data=data_to_write,
                    compression=compression,
                    compression_opts=compression_level,
                )
                f["kcoeff"].attrs["units"] = kdata_unit
            else:
                f.create_dataset(
                    "kcoeff",
                    data=self.kdata,
                    compression=compression,
                    compression_opts=compression_level,
                )
                f["kcoeff"].attrs["units"] = self.kdata_unit
            f.create_dataset("method", (1,), data=self.sampling_method, dtype=dt)
            f.create_dataset(
                "samples",
                data=self.ggrid,
                compression=compression,
                compression_opts=compression_level,
            )
            f.create_dataset(
                "weights",
                data=self.weights,
                compression=compression,
                compression_opts=compression_level,
            )
            f.create_dataset("ngauss", data=self.Ng)
            f.create_dataset(
                "bin_centers",
                data=self.wns,
                compression=compression,
                compression_opts=compression_level,
            )
            f.create_dataset(
                "bin_edges",
                data=self.wnedges,
                compression=compression,
                compression_opts=compression_level,
            )

            # where most of the data is actually written
            self.write_hdf5_common(
                f,
                compression=compression,
                compression_level=compression_level,
                p_unit=p_unit,
            )

    def bin_down(
        self,
        wnedges=None,
        weights=None,
        ggrid=None,
        remove_zeros=False,
        num=300,
        use_rebin=False,
        write=0,
    ):
        """Method to bin down a kcoeff table to a new grid of wavenumbers (inplace).

        Parameters
        ----------
            wnedges: array, np.ndarray
                Edges of the new bins of wavenumbers (cm-1)
                onto which the kcoeff should be binned down.
                if you want Nwnew bin in the end, wnedges.size must be Nwnew+1
                wnedges[0] should be greater than self.wnedges[0] (JL20 not sure anymore)
                wnedges[-1] should be lower than self.wnedges[-1]
            weights: array, np.ndarray, optional
                Desired weights for the resulting Ktable.
            ggrid: array, np.ndarray, optional
                Desired g-points for the resulting Ktable.
                Must be consistent with provided weights.
                If not given, they are taken at the midpoints of the array
                given by the cumulative sum of the weights
        """
        old_ggrid = self.ggrid
        if weights is not None:
            self.weights = np.array(weights)
            self.gedges = np.concatenate(([0], np.cumsum(self.weights)))
            if ggrid is not None:
                self.ggrid = np.array(ggrid)
            else:
                self.ggrid = (self.gedges[1:] + self.gedges[:-1]) * 0.5
            self.Ng = self.ggrid.size
        wnedges = np.array(wnedges)
        if wnedges.size < 2:
            raise TypeError("wnedges should at least have two values")
        wngrid_filter = np.where(
            (wnedges <= self.wnedges[-1]) & (wnedges >= self.wnedges[0])
        )[0]
        if not is_sorted(wnedges):
            raise RuntimeError("wnedges should be sorted.")
        indicestosum, wn_weights, wn_weight_counts = rebin_ind_weights(
            self.wnedges, wnedges[wngrid_filter]
        )
        if write > 10:
            print(indicestosum)
            print(wn_weights)

        self.kdata = bin_down_corrk(
            wnedges.size - 1,
            self.kdata,
            old_ggrid,
            self.ggrid,
            self.gedges,
            indicestosum,
            wngrid_filter,
            wn_weights,
            wn_weight_counts,
            num,
            use_rebin,
        )

        self.wnedges = wnedges
        self.wns = (wnedges[1:] + wnedges[:-1]) * 0.5
        self.Nw = self.wns.size
        if remove_zeros:
            self.remove_zeros(deltalog_min_value=10.0)


def rm_molec(unit_name):
    """Removes "/molecule" or "/molec" from a unit string.

    Parameters
    ----------
        unit_name: str
            String to be changed.

    Returns
    -------
        str
            The unit name without the ending "/molecule" or "/molec"
    """
    return unit_name.replace("/molecule", "").replace("/molec", "")


@jax.jit
def rebin(f_fine, fine_grid, coarse_grid):
    """Computes the binned version of a function on a coarser grid.
    The two grids do not need to have the same boundaries.

    Parameters
    ----------
        f_fine: array, np.ndarray
            Function to be rebinned, given on the fine_grid.
        fine_grid: array, np.ndarray
            The high resolution grid edges we start with.
        coarse_grid: array, np.ndarray
            The coarser resolution grid edges inside which we want to bin the function f.
    """
    fine_left = fine_grid[:-1][:, None]
    fine_right = fine_grid[1:][:, None]
    coarse_left = coarse_grid[:-1][None, :]
    coarse_right = coarse_grid[1:][None, :]
    interval_values = f_fine[: fine_grid.shape[0] - 1]

    overlap = jnp.maximum(
        0.0,
        jnp.minimum(fine_right, coarse_right) - jnp.maximum(fine_left, coarse_left),
    )
    weighted_sum = jnp.sum(overlap * interval_values[:, None], axis=0)
    total_overlap = jnp.sum(overlap, axis=0)
    safe_total_overlap = jnp.where(total_overlap > 0.0, total_overlap, 1.0)

    return jnp.where(total_overlap > 0.0, weighted_sum / safe_total_overlap, 0.0)


@partial(jax.jit, static_argnames=("new_nw", "num", "use_rebin"))
def bin_down_corrk(
    new_nw,
    kdata,
    old_ggrid,
    new_ggrid,
    gedges,
    indicestosum,
    wngrid_filter,
    wn_weights,
    wn_weight_counts,
    num,
    use_rebin,
):
    """bins down a kcoefficient table (see :func:`~exo_k.ktable.Ktable.bin_down` for details)

    Parameters
    ----------
        new_nw : int
            Number of wavenumber bins in the rebinned table.
        kdata : array, np.ndarray
            table to bin down.
        old_ggrid : array, np.ndarray
            Grid of old g-point abscissas for kdata.
        new_ggrid : array, np.ndarray
            New g-points for the binned-down k-coefficients.
        gedges : array, np.ndarray
            Cumulative sum of the weights. Goes from 0 to 1.
            Used only if use_rebin=True
        indicestosum: list of lists
            Indices of wavenumber bins to be used for the averaging
        wngrid_filter: array, np.ndarray
            Indices of the new table where there will actually be data (zero elsewhere)
        wn_weights: array, np.ndarray
            Padded weights to be used for the averaging.
        wn_weight_counts: array, np.ndarray
            Number of valid weights to use in each row of ``wn_weights``.
        num: int
            Number of points to fine sample the g function in log-k space
        use_rebin: boolean
            Whether to use rebin or interp method.
    """
    ggrid0to1 = jnp.asarray(old_ggrid).at[0].set(0.0).at[-1].set(1.0)
    dtype = jnp.asarray(kdata).dtype
    newkdata = jnp.zeros(
        (kdata.shape[0], kdata.shape[1], new_nw, new_ggrid.size),
        dtype=dtype,
    )
    contributor_offsets = jnp.arange(wn_weights.shape[1], dtype=indicestosum.dtype)
    interpolation_axis = jnp.linspace(0.0, 1.0, num, dtype=dtype)
    batch_size = kdata.shape[0] * kdata.shape[1]

    def interpolate_cdf(logkgrid_flat, tmp_logk_flat):
        def interpolate_batch(grid_row, logk_rows):
            return jax.vmap(
                lambda contributor_logk: jnp.interp(grid_row, contributor_logk, ggrid0to1)
            )(logk_rows)

        return jax.vmap(interpolate_batch)(logkgrid_flat, tmp_logk_flat)

    def body_fun(i_w, current_kdata):
        start_index = indicestosum[i_w] - 1
        contributor_count = wn_weight_counts[i_w]
        spectral_indices = jnp.minimum(
            start_index + contributor_offsets,
            kdata.shape[2] - 1,
        )
        weight_row = wn_weights[i_w]
        valid_mask = contributor_offsets < contributor_count
        tmp_logk = jnp.log(jnp.take(kdata, spectral_indices, axis=2))

        logk_min = jnp.min(
            jnp.where(valid_mask[None, None, :], tmp_logk[..., 0], jnp.inf),
            axis=2,
        )
        logk_max = jnp.max(
            jnp.where(valid_mask[None, None, :], tmp_logk[..., -1], -jnp.inf),
            axis=2,
        )
        is_constant = logk_min == logk_max

        expanded_logk_max = logk_max + (logk_max - logk_min) / (num - 3.0)
        expanded_logk_min = logk_min - (expanded_logk_max - logk_min) / (num - 3.0)
        safe_logk_min = jnp.where(is_constant, logk_min - 0.5, expanded_logk_min)
        safe_logk_max = jnp.where(is_constant, logk_max + 0.5, expanded_logk_max)

        logkgrid = safe_logk_min[..., None] + (
            safe_logk_max - safe_logk_min
        )[..., None] * interpolation_axis
        logkgrid_flat = logkgrid.reshape((batch_size, num))
        tmp_logk_flat = tmp_logk.reshape(
            (batch_size, contributor_offsets.size, kdata.shape[-1])
        )
        newg_flat = jnp.sum(
            interpolate_cdf(logkgrid_flat, tmp_logk_flat) * weight_row[None, :, None],
            axis=1,
        )
        exp_logkgrid_flat = jnp.exp(logkgrid_flat)

        if use_rebin:
            rebinned_flat = jax.vmap(
                lambda values, grid: rebin(values, grid, gedges)
            )(exp_logkgrid_flat, newg_flat)
        else:
            rebinned_flat = jax.vmap(
                lambda values, grid: jnp.interp(new_ggrid, grid, values)
            )(exp_logkgrid_flat, newg_flat)

        rebinned = rebinned_flat.reshape(
            (kdata.shape[0], kdata.shape[1], new_ggrid.size)
        )
        constant_values = jnp.broadcast_to(jnp.exp(logk_max)[..., None], rebinned.shape)
        rebinned = jnp.where(is_constant[..., None], constant_values, rebinned)

        return current_kdata.at[:, :, wngrid_filter[i_w], :].set(rebinned)

    return jax.lax.fori_loop(0, wngrid_filter.shape[0] - 1, body_fun, newkdata)


def rebin_ind_weights(old_bin_grid, new_bin_grid):
    """Computes the indices and weights to be used when
    rebinning an array of values f of size N in the bins delimited by old_bin_grid (of size N+1)
    onto new bins delimited by new_bin_grid (of size Nnew+1)
    Returns indices, padded weights, and the valid weight counts to be used as follows
    to proceed with the rebinning

    >>> f_rebinned = [
    ...     np.dot(
    ...         f[indicestosum[ii] - 1 : indicestosum[ii] - 1 + weight_counts[ii]],
    ...         final_weights[ii, : weight_counts[ii]],
    ...     )
    ...     for ii in range(new_bin_grid.size - 1)
    ... ]
    """
    #    new_bin_grid_used=np.where(new_bin_grid<old_bin_grid[0],old_bin_grid[0],new_bin_grid)
    indicestosum = np.searchsorted(old_bin_grid, new_bin_grid, side="right")
    if indicestosum[-1] == old_bin_grid.size:
        indicestosum[-1] -= 1
    dg = old_bin_grid[1:] - old_bin_grid[:-1]
    weight_counts = np.where(
        indicestosum[1:] > indicestosum[:-1],
        indicestosum[1:] - indicestosum[:-1] + 1,
        1,
    ).astype(np.int32)
    final_weights = np.zeros(
        (new_bin_grid.size - 1, np.max(weight_counts)),
        dtype=np.result_type(old_bin_grid.dtype, np.float64),
    )
    for ig in range(new_bin_grid.size - 1):
        if indicestosum[ig + 1] > indicestosum[ig]:
            weights = np.concatenate(
                (
                    old_bin_grid[[indicestosum[ig]]] - new_bin_grid[[ig]],
                    dg[indicestosum[ig] : indicestosum[ig + 1] - 1],
                    new_bin_grid[[ig + 1]] - old_bin_grid[[indicestosum[ig + 1] - 1]],
                )
            )
        else:
            weights = np.array([new_bin_grid[ig + 1] - new_bin_grid[ig]])
        weights = weights / np.sum(weights)
        final_weights[ig, : weight_counts[ig]] = weights
    return indicestosum.astype(np.int32), final_weights, weight_counts


@jax.jit
def is_sorted(a):
    """Finds out if an array is sorted. Returns True if it is."""
    return jnp.all(a[1:] >= a[:-1])


def unit_convert(
    quantity, unit_file="unspecified", unit_in="unspecified", unit_out="unspecified"
):
    """Chooses the final unit to use and the conversion factor to apply.

    Parameters
    ----------
        quantity: str
            The name of the pysical quantity handled for potential error messages
        unit_file/unit_in/unit_out: str

            Respectively:

              * String with the unit found in (or assumed from the format of) the initial data,
              * The unit we think the initial data are in if unit_file is 'unspecified'
                or (we believe) wrong,
              * The unit we want to convert to. If unspecified, we do not convert.
    Returns
    -------
        unit_to_write: str
            Resulting unit.
        conversion_factor: float
            A multiplicating factor for the data to proceed to the conversion.
    """
    if unit_in != "unspecified":
        if (unit_file != "unspecified") and (unit_file != unit_in):
            print("Be careful, you are assuming that " + quantity + " is " + unit_in)
            print(
                "but the input file says that it is "
                + unit_file
                + ". The former will be used"
            )
        starting_unit = unit_in
    else:
        if unit_file == "unspecified":
            raise NotImplementedError(
                """I could not find the {quantity} used in the file.
            So you should tell me what unit you think is used by specifying
            the keyword argument: file_{quantity}= a unit recognized by the astropy.units library.
            If you want to convert to another unit, add the {quantity} keyword
            with the desired unit.""".format(quantity=quantity)
            )
        else:
            starting_unit = unit_file
    if unit_out == "unspecified":
        unit_to_write = starting_unit
    else:
        unit_to_write = unit_out
    return unit_to_write, u.Unit(starting_unit).to(u.Unit(unit_to_write))
