# Change Log
All notable changes to the CCF module will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com)
and this project adheres to [Semantic Versioning](http://semver.org).

## [2.7.3] - 2023-11-08
### Changed
 - New plotting function interface, can pass `pRT_objects` and `model_generating_functions`
 - New MPI interface for better parallelisation
 - Fixed binning of spectra errorbars for plotting
 - Minor typo bug fixes

## [2.7.2] - 2023-11-08
### Fixed
 - Fixed incorrect MMW calculation in `chemistry.py`

## [2.7.1] - 2023-11-08
### Changed
 - Updated tick marks for log plots of spectra
 - Fixed bugs from 2.7.0 merge

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

### Removed
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
