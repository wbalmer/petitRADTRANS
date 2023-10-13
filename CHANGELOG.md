# Change Log
All notable changes to the CCF module will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org).

## [3.0.0-a80] - 2023-02-10
### Added
- SYSREM preparing pipeline.
- Simple transit light loss modelling for `SpectralModel`.
- Support for SLURM.
- Possibility to retrieve or optimize uncertainties.
- Trimming preparation function.
- Module `cli.prt_cli` to automatically download petitRADTRANS input data files.
- Module `configuration` to manage paths.
- Module `molar_mass`, as a more explicit interface between `petitRADTRANS` and `molmass`.
- Function `utils.feature_scaling` to normalize arrays.
- Function `utils.bayes_factor2sigma` to convert Bayes factor significance into "sigma" significance.
- Function `SpectralModel.resolving_space` to generate arrays with values spaced at constant resolving power.
- Constant `e_molar_mass` to physical constants.
- Some helpful error and warning messages.

### Changed
- Functions, arguments and attributes now have clearer names and respect PEP8.
- Spectral functions of `Radtrans` (`calculate_flux` and `calculate_transit_radii`) now return wavelengths, spectrum, and a dictionary containing additional outputs, instead of nothing.
- Function `Radtrans.calculate_flux` now output by default wavelengths in cm (instead of frequencies in Hz) and flux in erg.s-1.cm-2/cm instead of erg.s-1.cm-2/Hz. Setting the argument `frequencies_to_wavelengths=False` restores the previous behaviour.
- Function `Radtrans.calculate_transit_radii` now output by default wavelengths in cm (instead of frequencies in Hz). Setting the argument `frequencies_to_wavelengths=False` restores the previous behaviour.
- Improved memory usage of object `Radtrans`.
- Object `Radtrans` is now imported using `from petitRADTRANS.radtrans import Radtrans` (was `from petitRADTRANS import Radtrans`) for more stable installation.
- Module `nat_cst` renamed `physical_constants`.
- Some functions have moved from the module `physical_constants` to another, more specific module.
- Some functions have moved from the module `Radtrans` to another, more specific module.
- Input data path is now stored in a config file within the folder \<HOME\>/.petitRADTRANS, generated when installing the package or using it for the first time.
- Attribute `SpectralModel.times` is now inside `SpectralModel.model_parameters`.
- Function `preparing_pipeline` now only masks invalid points instead of the entire column/line where the point was.
- In `SpectralModel`, orbital longitudes and radial velocity semi-amplitudes are know calculated instead of fixed.
- Rules for opacites and species names are now clearly defined, based on the ExoMol format.
- Structure of directory input_data now is akin to ExoMol.
- Line-by-line opacities can now be read from HDF5 files.
- Cloud opacities can now be read from HDF5 files.
- CIA cross-sections can now be read from HDF5 files.
- petitRADTRANS is now installed through `meson` instead of the deprecated `numpy.distutils`. The installation procedure is mostly unchanged.
- Various optimisations.
- Package structure.
- Code clean-up.

### Removed
- Multiple `Radtrans` attributes, some are now function outputs.
- Class `ReadOpacities`, now merged with `Radtrans`.
- Module `pyth_input`, now merged with `Radtrans`.
- Module `version`, version is now defined in pyproject.toml.
- Deprecated `molecular_weight` constant.

### Fixed
- Re-binning correlated-k opacities no longer require to re-launch petitRADTRANS.
- Oscillating telluric lines depth when generating shifted and re-binned mock observations with `SpectralModel`.

## [2.6.0] - 2023-03-27
### Added
- High-resolution retrievals.
- Possibility to initialize a `retrieval.Data` class without needing a file.
- Possibility to initialize a `retrieval.Data` class with a `Radtrans` object, without the need to re-create one.
- Possibility to initialize a `retrieval.Retrieval` class with a stellar spectrum, without the need to recalculate it.
- Possibility to give scattering.
- Support for 2D and 3D spectral array in retrievals.
- Static function `retrieval.Retrieval._get_samples`, to get retrieval outputs without the need to initialize a `Retrieval`.
- Gibson et al. 2021 log-likelihood calculation.
- Better high-resolution mock observation function.
- Module `phoenix` for PHOENIX stellar models.
- Module `physics` to store useful physical functions.
- Module `utils` to store generic useful functions.
- Module `retrieval.preparing` to prepare telluric-contaminated ground-based data.
- Module `ccf` to perform cross-correlation analysis.
- Module `cli.eso_etc_cli` as a custom interface to ESO's ETC CLI.
- Module `cli.eso_skycalc_cli` as a custom interface to ESO's SKYCALC CLI.
- Class `SpectralModel` as a wrapper to `Radtrans` and `Retrieval` objects.
- Function `get_guillot_2010_temperature_profile`, a more general Guillot 2010 temperature profile.
- Function to calculate the ESM of a planet.
- Function to calculate the orbital phase of a planet.
- Function to calculate the radial velocity of a planet.
- Function to calculate the orbital velocity of a planet.
- Function to calculate the Doppler shift.
- Function to convolve, run-convolve, Doppler shift, and rebin a spectrum.
- Noise estimation for eclipse spectra in addition to transit spectra.
- Method to generate a `Planet` object using a NASA Exoplanet Archive tab file.
- Function to calculate the radius of a planet from its surface gravity and mass.
- Option to use the NASA Exoplanet Archive "best mass" when generating a `Planet`.
- Functions in `Radtrans` to calculate radius and pressure at hydrostatic equilibrium.
- Na2S and KCl clouds compatibility with free chemistry.
- [Future] Module `configuration` to manage paths.
- Module `version` to store petitRADTRANS version number (will be used in future version).
- Message when loading the `poor_mans_nonequ_chem` chemical equilibrium mass mixing ratios table.

### Changed
- Running mean now uses the faster `scipy.ndimage.filters.uniform_filter1d` implementation.
- Some cloud functions are more generic.
- Character limit in retrieval output directory increased from 100 to 200.
- Stricter scattering convergence criterion.
- Switched to corr-k combination method without Monte Carlo noise (relevant for scattering mode).
- CIAs are no more hard-coded.
- Make `poor_mans_nonequ_chem` compatible with `[Future] Orange`.
- Code clean-up.

### Removed
- Useless make files.

### Fixed
- Hansen cloud particle distribution returning NaN if `b_hansen` set too low.
- Retrieval not converging when using correlated-k.
- Crash when trying to read a nonexistent opacity file.
- Function `contour_corner` not working when not giving optional arguments `parameter_ranges` and `parameter_plot_indices`.
- True values not plotted in function `contour_corner`.
- Function `getMM` not working with e- and H-.
- e- and H- implementation.
- Hack cloud photospheric tau behaviour.
- Potential reference before assignment in module `retrieval`.
- Potential `TypeError` in `model_generating_function`.
- Module `setup` not working with PyPI.
- Wavelength range in module `retrieval` not working in "photometry" mode.
- Wrong docstrings in function `Radtrans.get_star_spectrum`.
- Argument `add_cloud_scat_as_abs` being `None` by default instead of `False`.


---
No changelog before version 2.6.0.
Fixing updates of versions 2.6.x not tracked.
Some additions, changes and fixes reported in 2.6.0 may have been implemented in previous versions.
Test suite added in version 2.4.0.
