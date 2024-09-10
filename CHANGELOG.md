# Change Log
All notable changes to petitRADTRANS will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org).

## [3.1.0a42] - 2024-09-10
### Added
- Equilibrium mass fraction support for SiO clouds in `chemistry.clouds`.
- Full integration of partial cloud coverage in `Radtrans`.
- Patchy clouds can now be applied to individual cloud components, rather than only fully clear and cloudy, using the `remove_cloud_species` parameter (use full name).
- Function to convolve a spectrum with a variable width kernel, based on Brewster implementation. Can be used in a retrieval if the `data_resolution` parameter is set as an array.
- Emission models for retrievals can now include a simple circumplanetary disk model, given blackbody temperature and disk radius parameters.
- Functions `plot_result_corner` and `contour_corner` can now use all the functionalities of the [`corner.corner` function](https://corner.readthedocs.io/en/latest/).
- Possibility to generate mock input data for input == output retrievals, using the exact same format as the input data.
- Possibility to run pRT's retrieval model with emcee (base implementation with less functionality than the full retrieval package, i.e., no plotting support for result analysis)
- Possibility to use vertically variable fsed per species.
- Possibility to custom `SpectralModel` spectral modification functions at instantiation.
- Possibility to load any crystalline cloud opacities without giving the space group if there is only one space group available for this cloud species.
- Possibility to specify the retrieval name in `plot_result_corner`.
- Possibility to load line-by-line opacities with different frequency grid boundaries.
- Genericised temperature gradient profile function `dtdp_temperature_profile` to accept different top/bottom of atmosphere pressures.
- Function to get a forward model(s) of a retrieval, with option to get the best fit model or the Xth quantile model.
- Function to output opacity contribution spectra for `Radtrans` and `SepctralModel` objects.
- Function to plot the above opacity contribution spectra.
- Function to estimate atmospheric metallicity and element-to-hydrogen ratios from mass fractions.
- Function to fill all layers of an atmosphere at once with filling species.
- Functions to convert frequencies into wavelengths (in cm or in um), and vice-versa.
- Function `Radtrans.get_wavelengths` to obtain the equivalent in cm of a `Radtrans` object's frequency grid.
- Function to directly get retrieval samples into a dict.
- Function to convert the default log-likelihood to a chi2.
- Documentation for the newly added functions.
- Warnings for negative temperature, mass fractions, and mean molar masses when calculating opacities.
- Warning when only one of the two parameters necessary to include a power law opacity has been set.
- Test module for `SpectralModel` using custom functions.
- Test module for `SpectralModel` in `'c-k'` opacity mode.
- Test module for `SpectralModel` in `'lbl'` opacity mode.
- Test module for `SpectralModel` retrieval framework.
- Performance tests.
- Source files for JOSS papers.

### Changed
- Future: parameter `emission_geometry` is canonically renamed `irradiation_geometry`. The old parameter will be deprecated in version 4.0.0.
- Future: key `'modification_parameters'` of `SpectralModel.model_parameters` is canonically renamed `'modification_arguments'`. The old key will be deprecated in version 4.0.0.
- Clarified a bit the documentation on the `SpectralModel` retrieval framework.
- Requested input and output parameter names for externally provided function to load opacities for `format2petitradtrans`: since cm^2 should be returned it should be called cross-sections, not opacities.
- Restructured `retrieval.models.py` to reduce code reuse. New functions to generically compute an emission or transmission spectrum with patchy clouds.
- Previous 'patchy' model functions are now redundant, all the functions can accept the same patchy cloud parameters. Still included for backwards compatibility.
- The `examples` directory is relocated to the notebook directory, and renamed `retrievals`.

### Removed
- Unused test functions.
- Example retrieval output files.

### Fixed
- Bug in `retrieval.loglikelihood()` where offsets in datasets were not applied if `external_radtrans_reference` was not `None`.
- Bug in function `retrieval.plot_spectra()` when plotting the best-fit spectrum together with `radtrans_grid=True`.
- Bug in function `calculate_transit_radii()` when `return_opacities=True`.
- Function `format2petitradtrans` applied the incorrect pRT wavelength grid to the lbl opacity conversion.
- Function `rebin_spectrum_bin` incorrectly handling overlapping bins.
- Crash when unpickling `LockedDict` objects.
- Crash when loading unspecified source opacities with different spectral info than the default opacity file and multiple files with that spectral info exist.
- Crash of `SpectralModel` when adding the transit light loss effect without shifting the spectrum.
- Crash of `SpectralModel` when adding a star spectrum on shifted spectra.
- Function `Retrieval.plot_spectra` not working when `mode='median'`.
- Mass fractions being modified when calculating CIA opacities in some cases.
- Cloud mass fractions are taken into account when filling atmosphere.
- Electron symbol (`'e-'`) not supported as a `SpectralModel` imposed mass fraction.
- Crash of `SpectralModel` when not specifying the mass fraction of a line species.
- Crash when preparing fully masked spectra.
- Crash when using a fresh `SpectralModel` instance's `calculate_spectrum` with `update_parameters=False` without initializing `star_flux`.
- Crash when negative data are used for a retrieval.
- Bug in patchy cloud implementation. For pRT3, clouds in abundance dict are now addressed using full name.
- Model functions lacking the required 350nm scattering parameter for hazes as an optional parameter (solves [issue 76](https://gitlab.com/mauricemolli/petitRADTRANS/-/issues/76)).
- Fixes to `madhushudhan_seager_transmission` function (solves [issue 80](https://gitlab.com/mauricemolli/petitRADTRANS/-/issues/80)). 
- Crash due incorrect shape of sample arrays (solves [issue 82](https://gitlab.com/mauricemolli/petitRADTRANS/-/issues/82)). 
- Crash when trying to get the samples of a retrieval with one live point.
- Out-of-memory errors when converting large opacity files on systems with 16 GB of RAM or less.
- Incorrect filling mass fraction calculation in some cases.
- Opacities may be loaded from incorrect source if the source's name is included in another opacity's source name (e.g. 'Allard' and 'NewAllard').
- Unable to automatically download a default opacity file.
- Silent error when calculating the transit effect for a non-transiting planet.
- Function maps of `SpectralModel` are incorrectly loaded.
- Bug in emission retrieval tutorial, changing names of cloud parameters.
- Thulium (Tm), Americium (Am), Curium (Cm) and Fermium (Fm) are identified as negatively charged species.
- Incorrect behaviour: during the preparing step, data and uncertainties with inconsistent masks are tolerated.
- Incorrect behaviour: mass fractions species names without spectral info are not recognized if opacities species names have spectral info (loading opacities with different spectral info is not possible anyway).
- Incorrect behaviour: `exo_k` is imported to bin down opacities even when the binned-down opacity file already exists.
- Typos in some docs.
- Typos in some comments.

### Pending
- Temporarily reverted to allow < 0 solutions in the tridiagonal solver until it is determined if they should be allowed.
- Temporarily silented the overflow warning message until a solution to trigger the message less often is found.
- Temporarily set clouds space group to their undefined value (`000`) until their actual space group is found.

## [3.0.7] - 2024-07-01
### Fixed
- Fixed log-likelihood bug from change in sample array shape [(issue 71)](https://gitlab.com/mauricemolli/petitRADTRANS/-/issues/71). All internal sample arrays should now have shape (number_params,number_samples).
- Fixed corner plot memory bug.
- Updated contribution plots and abundance plots for pRT3.

## [3.0.6] - 2024-06-18
### Fixed
- Fixed additional crash when using H- opacities.

## [3.0.5] - 2024-06-12
### Fixed
- Crash when including H- opacities.

## [3.0.4] - 2024-05-28
### Fixed
- Scripts hang forever while loading `exo-k` re-binned opacities on multiple processes.
- Indices for AMR in retrievals are not integers.

## [3.0.3] - 2024-05-15
### Fixed
- Fixed default `haze_factor` value in `retrieval/models.py`.

## [3.0.2] - 2024-05-07
### Fixed
- Electron symbol (`'e-'`) not supported as a `Radtrans` mass fraction.

## [3.0.1] - 2024-05-02
### Fixed
- Crash of DACE opacities conversion when not using the debug wavelength grid file.
- Crash of DACE opacities conversion when converted opacities have an insufficient wavenumber coverage.
- Crash when loading line-by-line opacities using wavenumber grids with insignificant differences (i.e. < 1e-12 relative differences).

## [3.0.0] - 2024-04-29
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
- Function that generates the petitRADTRANS default wavelength grid.
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
- Temporarily allowed < 0 solutions in function `fortran_radtrans_core.math.solve_tridiagonal_system` until it is determined if they should be allowed.
- Temporarily silented the overflow warning message of function `fortran_radtrans_core.math.solve_tridiagonal_system` until a solution to trigger the message less often is found.
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

## [2.7.7] - 2024-03-06
Should be 2.9.0 (no code change since last version).
### Changed
 - Various documentation updates.

## [2.7.6] - 2023-11-28
Should be 2.9.0.
### Added
- Implementation of leave-one-out cross validation retrieval from Sam de Regt.
- Includes new Pareto Smoothed INP and LOO module (psis.py), plus additional functions in retrieval module.
- Example included in test directory.

## [2.7.5] - 2023-11-08
Should be 2.8.2.
### Changed
 - Updates to retrieval tutorials, as requested by JOSS review.

## [2.7.4] - 2023-11-13
Should be 2.8.1.
### Changed
 - Minor changes to MPI interface.
 - Removed unnecessary instructions for installation on Apple Mx chips.
 - Updated interface to `species`, now requiring newer than v0.7.

### Fixed
 - Bugs in argument naming.

## [2.7.3] - 2023-11-08
Should be 2.8.0.
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
