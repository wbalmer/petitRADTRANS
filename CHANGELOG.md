# Change Log
All notable changes to petitRADTRANS will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org).

## [3.0.0-a173] - 2024-04-16
### Added
- Automatic download of missing input_data files.
- Automatic binning-down of correlated-k opacities when instantiating a `Radtrans` object.
- Simple transit light loss modelling for `SpectralModel`.
- Possibility to retrieve or optimize uncertainties.
- Trimming preparation function.
- Module `cli.prt_cli` to automatically download petitRADTRANS input data files.
- Module `configuration` to manage paths.
- Module `plotlib` to store useful plot functions and plot styles.
- Module `stallar_spectra`, to handle stellar spectra calculations.
- Function `utils.feature_scaling` to normalize arrays.
- Function `utils.bayes_factor2sigma` to convert Bayes factor significance into "sigma" significance.
- Function `SpectralModel.resolving_space` to generate arrays with values spaced at constant resolving power.
- Function `chemistry.utils.fill_atmospheric_layer` to fill an atmospheric layer using weighted filling species.
- Function `retrieval.retrieval.save_best_fit_outputs_external_variability` to speed up calculations, since it makes use of `external_prt_reference` properly.
- Constant `e_molar_mass` to physical constants.
- SysRem preparing pipeline.
- Arguments of `SpectralModel.calculate_spectrum` in `model_parameters`.
- MgFeSiO4 cloud mass fraction calculation.
- Object `SpectralModel` support for models of any complexity.
- More control over uncertainty_scaling_b: can now also be done per observation "types" (e.g., different MRS channels see the same b).
- Possibility for the forward model to return flux of different atmospheric columns, which are then mixed with the atmospheric_column_flux_mixer function given to the data object, depending on the data's epoch and exposure time.
- Possibility to return only the best fit spectra are calculated and no plot is produced.
- Possibility to return the gases and clouds opacities calculated by the `Radtrans` object.
- Possibility to initialize a `Radtrans` object with CIA opacities, cloud opacities, or no opacities.
- Possibility to print the log-likelihood to the console if desired, for retrieval debugging.
- Possibility for function `retrieval.retrieval.save_best_fit_outputs` to return the best fit spectra.
- Possibility to set the clouds particles porosity factor.
- Treatment to allow for concatenated fluxes (from different epochs) for variability retrievals.
  (see `concatenate_flux_epochs_variability` and its use in the data class).
- Handling of column flux mixing in `retrieval.retrieval.save_best_fit_outputs`.
- Some helpful error and warning messages.
- Argument `seed` to function `Retrieval.run`, for reproducibility.
- Better test suite workflow.
- Module `docs/utils` to store functions useful to build the docs.

### Changed
- Added column_flux_mixer treatment to petitRADTRANS/retrieval/data.py's get_chisq() in convolve-rebin mode.
- TODO: `fortran_radtrans_core.math.solve_tridiagonal_system`:
  - temporarily reverted to allow < 0 solutions in the tridiagonal solver until it is determined if they should be allowed.
  - temporarily silented the overflow warning message until a solution to trigger the message less often is found.
- Functions, arguments and attributes now have clearer names and respect PEP8. The complete list of change is available [here](https://docs.google.com/spreadsheets/d/1yCiyPJfUXzsd9gWTt3HdwntNM2MrHNfaZoacXkg1KLk/edit#gid=2092634402).
- Spectral functions of `Radtrans` (`calculate_flux` and `calculate_transit_radii`) now return wavelengths, spectrum, and a dictionary containing additional outputs, instead of nothing.
- Function `Radtrans.calculate_flux` now output by default wavelengths in cm (instead of frequencies in Hz) and flux in erg.s-1.cm-2/cm instead of erg.s-1.cm-2/Hz. Setting the argument `frequencies_to_wavelengths=False` restores the previous behaviour.
- Function `Radtrans.calculate_transit_radii` now output by default wavelengths in cm (instead of frequencies in Hz). Setting the argument `frequencies_to_wavelengths=False` restores the previous behaviour.
- Complete rework of the input_data file naming convention.
- The combination of correlated-k opacities are now using the faster merge sorting algorithm.
- Significantly improved the CIA interpolation performances. This might cause small changes in some results (most of the time the relative change should be < 1e-6).
- Significantly improved transmission spectra calculation performances.
- Significantly improved the `polyfit` preparing pipeline performances by using `numpy.polyfit` instead of the recommended `numpy.polynomial.Polynomial.fit`.
- Slightly improved Feautrier radiative transfer calculation performances (`calculate_flux`).
- Improved memory usage of object `Radtrans`.
- Object `Radtrans` is now imported using `from petitRADTRANS.radtrans import Radtrans` (was `from petitRADTRANS import Radtrans`) for more stable installation.
- Object `SpectralModel` is now imported using `from petitRADTRANS.spectral_model import SpectralModel`.
- Object `Planet` is now imported using `from petitRADTRANS.planet import Planet`.
- Module `nat_cst` renamed `physical_constants`.
- Module `poor_mans_nonequ_chem` renamed `chemistry.pre_calculated_chemistry` and reworked.
- Module `phoenix` renamed `stellar_spectra.phoenix` and reworked.
- Some functions have moved from the module `physical_constants` to another, more specific module.
- Some functions have moved from the module `Radtrans` to another, more specific module.
- Input data path is now stored in a config file within the folder \<HOME\>/.petitRADTRANS, generated when installing the package or using it for the first time.
- Attribute `SpectralModel.times` is now inside `SpectralModel.model_parameters`.
- Function `polyfit` now only masks invalid points instead of the entire column/line where the point was.
- In `SpectralModel`, orbital longitudes and radial velocity semi-amplitudes are know calculated instead of fixed.
- In function `retrievals.data.get_chisq`, variable `scale_err` is now applied after 10^b error scaling, not before.
- Rules for opacites and species names are now clearly defined, based on the ExoMol format.
- Structure of directory input_data now is akin to ExoMol.
- Line-by-line opacities are now read from HDF5 files.
- Cloud opacities are now read from HDF5 files.
- CIA cross-sections are now read from HDF5 files.
- petitRADTRANS is now installed through `meson` instead of the deprecated `numpy.distutils`. The installation procedure is mostly unchanged.
- Various optimisations.
- Updated package structure.
- Code clean-up.

### Removed
- Space group info for cloud species in _get_base_cloud_names().
- Multiple `Radtrans` attributes, some are now function outputs.
- Function `get_radtrans` of object `SpectralModel`, as `SpectralModel` is now a child of `Radtrans`.
- Class `ReadOpacities`, now merged with `Radtrans`.
- Module `pyth_input`, now merged with `Radtrans`.
- Module `version`, version is now defined in pyproject.toml.
- Deprecated `molecular_weight` constant.

### Fixed
- Added again the porosity density decrease for DHS.
- Crash when using photospheric cloud with null mass fractions.
- Bug in retrieval model that would break the log-likelihood calculation in case of an external_prt_reference.
- Bug in function `plot_radtrans_opacities` (TODO: what was it?).
- Re-binning correlated-k opacities requires to re-launch petitRADTRANS.
- Function `chemistry.volume_mixing_ratios2mass_fractions` always returning an empty dict.
- Zero opacity values in k-tables creating NaNs when using `exo-k` to bin them down.
- Emission scattering mode impacting transmission spectra (slight change in the result).
- Oscillating telluric lines depth when generating shifted and re-binned mock observations with `SpectralModel`.
- Too strict relevance threshold when combining correlated-k opacities.
- Debugging text displayed in the `Radtrans` photospheric radius calculation.
- Incorrect behaviour: importing the chemical table triggers its loading.
- Incorrect behaviour: importing the PHOENIX stellar spectra table triggers its loading.
- Incorrect behaviour: plotting when running in "evaluate" mode of the retrieval package.

## [2.9.0] - 2023-11-28
Referred as 2.7.6
### Added
- Implementation of leave-one-out cross validation retrieval from Sam de Regt.
- Includes new Pareto Smoothed INP and LOO module (psis.py), plus additional functions in retrieval module.
- Example included in test directory.

## [2.8.2] - 2023-11-08
Referred as 2.7.5
### Changed
 - Updates to retrieval tutorials, as requested by JOSS review.

## [2.8.1] - 2023-11-13
Referred as 2.7.4
### Changed
 - Minor changes to MPI interface.
 - Removed unnecessary instructions for installation on Apple Mx chips.
 - Updated interface to `species`, now requiring newer than v0.7.

### Fixed
 - Bugs in argument naming.

## [2.8.0] - 2023-11-08
Referred as 2.7.3
### Added
- Plotting function interface, can pass `pRT_objects` and `model_generating_functions`.
- MPI interface for better parallelisation.

### Fixed
 - Fixed binning of spectra errorbars for plotting
 - Minor typo bug fixes

## [2.7.2] - 2023-11-08
### Fixed
 - Incorrect MMW calculation in `chemistry.py`

## [2.7.1] - 2023-11-08
### Fixed
 - Tick marks for log plots of spectra
 - Bugs from 2.7.0 merge

## [2.7.0] - 2023-09-12
### Added
 - New model_generating_functions in `models.py`.
 - New functions in `retrieval.py` to access chemical abundance profiles.
 - beta: interface retrievals with easychem
 - Complete implementation of scaling factors for flux errors, line b scaling and offsets.
 - Mixed equilibrium and free chemistry retrievals
 - New temperature profiles (spline, gradient), with curvature prior
 - Can get Teffs from posterior distributions.
 - Can get volume mixing ratios from posterior distributions
 - Can retrieve haze scattering slopes.

### Changed
 - Can now generally set the contribution function, reference pressure, H2 and He abundances as parameters.
 - Can now plot VMRs as well as mass fractions.
 - Can use data in the format [wlen, bins, flux, error]
 - Many small improvements to speed throughout model and plotting routines.
 - More thorough cloud implementations - hansen, log normal, single radius, and more.

### Fixed
 - Calculation of chi2 - now automatically outputs normalisation by ndata and dof.
 - Corrected implementation of free cloud base pressures
 - Consistent binning and convolution in spectra plots

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
- Function `get_MM` not working with e- and H-.
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
Some additions, changes and fixes reported in 2.6.0 may have been implemented in previous versions.
Test suite added in version 2.4.0.
