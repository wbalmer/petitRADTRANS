.. petitRADTRANS documentation master file, created by
   sphinx-quickstart on Tue Jan 15 15:07:15 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==============================================
petitRADTRANS: exoplanet spectra for everyone!
==============================================

Welcome to the **petitRADTRANS** (pRT) documentation. pRT is a Python package for calculating spectra of exoplanets and for running retrievals.

In a nutshell, pRT allows you to calculate planetary transmission or emission spectra (including multiple scattering). For this you can make use of pRT's extensive opacity database (including gas and cloud opacities). pRT's built-in retrieval package allows you to run retrievals that combine data sets of varying resolution, wavelength coverage and atmospheric contexts (every data set may use its own forward model).

**To get started with some examples on how to run pRT immediately, see** `"Getting started" <content/notebooks/getting_started.html>`_.
**Otherwise read on for some more general information, including pRT's feature list.**

.. important:: **This is the documentation of petitRADTRANS, Version 3.**
   Version 3 comes with a lot of quality-of-life features, optimization, and code rationalization, additional opacities, but also with breaking changes.
   See the `pRT3 changes page <content/pRT3_changes_description.html>`_ for a summary of the pRT 3 improvements and changes. If you are a Version 2 user, we strongly recommend to switch to Version 3. We explain in the `adding opacities section <content/adding_opacities.html>`_ how your opacity database can be converted to Version 3. This is only relevant if you use non-standard pRT opacities that you calculated or converted yourself. That being said, Version 2 can still be installed via ``pip install petitRADTRANS==2.7.7``, and the old documentation is available in the `here <https://petitradtrans.readthedocs.io/en/2.7.7/>`_. **But please note that Version 2 will not be maintained any longer.**

pRT feature list
================

For each feature listed below, there is a detailed demonstration in the tutorial; just click on the links in each feature paragraph.

- **Spectra** -- pRT allows you to calculate transmission spectra (``calculate_transit_radii()``) and emission spectra (``calculate_flux()``) via the ``Radtrans`` object. The basic usage of these methods is demonstrated in `"Getting started" <content/notebooks/getting_started.html>`_.
- **Clouds with pRT** -- pRT allows you to to incorporate the effects of clouds in many different ways (e.g., gray clouds, power law clouds, clouds from optical constants with `EddySed <https://ui.adsabs.harvard.edu/abs/2001ApJ...556..872A/abstract>`_, or your own cloud prescription). This is demonstrated in `"Including clouds" <content/notebooks/including_clouds.html>`_.
- **Scattering** -- scattering is always included as an extinction process when calculating transmission spectra with pRT. For emission spectra, multiple scattering is added when requested by the user (it increases the runtime). Radiation is then scattered both in the atmosphere and on the planetary surface, if present. See `"Scattering for Emission Spectra" <content/notebooks/scattering_for_emission_spectra.html>`_ for a demonstration.
- **Atmospheric composition** -- to calculate a spectrum of an atmosphere you first have to know its composition. pRT comes with a method to interpolate chemical equilibrium compositions as a function of atmospheric metallicity, C/O, temperature, and pressure, including a simple quench treatment to mimic disequilbrium chemistry. See `"Interpolating chemical equilibrium abundances" <content/notebooks/interpolating_chemical_equilibrium_abundances.html>`_ for a demonstration. If a species is not listed in the equilibrium table, or if you want to use an elemetal composition that is different from a scaled solar composition, you can use our stand-alone `easyCHEM <https://easychem.readthedocs.io/en/latest/>`_ package.
- **Retrievals** -- pRT comes with a retrieval package that allows you to easily fit exoplanet spectra. It is straightforward to combine multiple data sets of various wavelength coverages and resolutions, define custom forward models per data set (or make use of our `pre-implemented forward model list <content/notebooks/retrieval_models.html>`_), define fixed and free parameters, using various priors (uniform, log-uniform, Gaussian, or your own choice!). Retrievals are run with `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest/>`_ (`Buchner et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014A%26A...564A.125B/abstract>`_), which needs to be installed separately, see `"Installation" <content/installation.html#prerequisite-for-retrievals-multinest>`_. An introduction on running retrievals with pRT can be found at `"Retrieval Examples" <content/retrieval_examples.html>`_.
- **Spectral model class** -- the ``SpectralModel`` object is a child of pRT's base ``Radtrans`` object and offers many convenience features, especially for calculating high-resolution spectra and for retrievals. While pRT's standard usage requires the user to specify the atmospheric state (temperature, abundance, cloud structure) "by hand", ``SpectralModel`` allows you to do the same in a quicker and more organized way. ``SpectralModel`` comes with predefined models for the atmospheric states (e.g., to calculate the temperatures profile, mass fractions, etc.) that can be fully customized. It is also straightforward to run retrievals with ``SpectralModel``, see `here <content/notebooks/retrieval_spectral_model.html>`_. While standard pRT is explicit ("what you see is what you get"), ``SpectralModel`` is more implicit, but can save you many lines of code and eases the model construction process. Its usage is demonstrated in `"Specral Model" <file:///Users/molliere/Documents/Project_Docs/petitRADTRANS/docs/_build/html/content/notebooks/spectral_model.html>`_.
- **Gas opacity treatment** -- gas opacities in pRT are treated via two different modes, the low resolution mode runs calculations at :math:`\lambda/\Delta\lambda\leq 1000` using the so-called correlated-k treatment (``line_opacity_mode='c-k'``) for opacities. The high resolution mode runs calculations at :math:`\lambda/\Delta\lambda\leq 10^6`, using a line-by-line opacity treatment (``line_opacity_mode='lbl'``). Opacities can be easily binned to lower resolution in ``'c-k'`` mode, or down-sampled in ``'lbl'`` mode. See `"Rebinning opacities" <content/notebooks/rebinning_opacities.html>`_ for a demonstration. Which mode to pick depends on what you want to use pRT for. For example, in a retrieval your model resolution should be high enough to allow for an accurate representation of the data, but not higher (higher resolutions make pRT run less fast). In terms of pRT usage, the two opacity modes ``'lbl'`` and ``'c-k'`` only differ in terms of available opacities (except for the differing binning vs. down-sampling treatment).
- **Opacity database** -- pRT comes with an extensive opacity database, described in `"Available opacities" <content/available_opacities.html>`_. You don't need to download a huge opacity input folder when installing pRT; requested opacities will be downloaded from our `Keeper opacity server <https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0/>`_ by pRT on-the-fly if requested during pRT initialization. If that does not work (e.g., your HPC cluster restricts how the internet can be accessed): the aforementioned Keeper link also allows you to access opacity files manually, and to move them into pRT's opacity folder as explained, for example, in the Exomol section of `"Adding opacities" <content/adding_opacities.html#importing-opacity-tables-from-the-exomol-website>`_. We also describe `how to add opacities <content/adding_opacities.html>`_ that may be missing from our database. For the easiest cases this may correspond to simply dropping a file which was downloaded from an external database into the pRT opacity folder.
- **Utility functionalities** -- pRT comes with additional functionalities to help you speed up your work.

   - The ``Planet`` class offers you a powerful way to access planetary parameters from `NASA's Exoplanet Archive <https://exoplanetarchive.ipac.caltech.edu/index.html>`_. You can also use it to calculate various quantities such as barycentric and orbital velocities, airmasses, equilibrium temperatures, ... Its use is demonstrated in the `Planet notebook <content/notebooks/planet.html>`_.
   - You can calculate contribution functions for both `emission <content/notebooks/analysis_tools.html#Emission-contribution-functions>`_ and `transmission <content/notebooks/analysis_tools.html#Transmission-contribution-functions>`_ spectra.
   - You can plot opacities of gaseous line absorbers, see the `analysis tools notebook <content/notebooks/analysis_tools.html#Plotting-opacities>`_.
   - Natural constants (mainly from `astropy <https://www.astropy.org/>`_ and `scipy <https://scipy.org/>`_ are available in pRT's internal unit system (cgs), see the `utility function notebook <content/notebooks/utility_functions.html#Constants>`_.
   - The Planck function :math:`B_\nu(T)` (see the `Planck function section <content/notebooks/utility_functions.html#Planck-function>`_) and a grid of synthetic host star spectra is available (see the `PHOENIX section <content/notebooks/utility_functions.html#PHOENIX-and-ATLAS9-stellar-model-spectra>`_).
   - The Guillot temperature model is implemented, (see the `Guillot section <content/notebooks/utility_functions.html#Guillot-temperature-model>`_).

License and Attribution
=======================

petitRADTRANS is available under the MIT License.

The following papers document pRT:

- The base package is described in `Mollière et al. (2019) <https://arxiv.org/abs/1904.11504>`_.
- The self-scattering implementation (relevant for, e.g., cloudy self-luminous planets) is described in `Mollière et al. (2020) <https://arxiv.org/abs/2006.09394>`_.
- The stellar light and surface scattering is desccribed in `Alei et al. (2022) <https://arxiv.org/abs/2204.10041>`_.
- The retrieval package is documented in `Nasedkin et al. (2024) <https://doi.org/10.21105/joss.05875>`_.

Please cite the base paper and those relevant to your work if you make use of petitRADTRANS.

Active Developers
=================

- Doriann Blain (main developer, and mastermind behind Version 3)
- Paul Mollière
- Evert Nasedkin (mastermind behind the retrieval package)

Since publishing the first paper on pRT in 2019, the lead development has been passed on from Paul Mollière to other members of the team, listed alphabetically above. At the moment, Doriann Blain is the lead developer.
Right now, Paul's main job is to determine the strategic directions for the code, managing its development, and adding new features here and there, from time to time...

We also welcome new contributors!
If you would like to become a contributor, please see `"Contributing" <content/contributing.html>`_.

Former contributors and developers
==================================

- Eleonora Alei
- Karan Molaverdikhani
- Tomas Stolker
- Nick Wogan
- Mantas Zilinskas

.. toctree::
   :maxdepth: 2
   :caption: Guide

   content/installation
   content/pRT3_changes_description
   content/tutorial
   content/available_opacities
   content/retrieval_examples
   content/adding_opacities
   content/contributing

.. toctree::
   :maxdepth: 2
   :caption: Code documentation

Contact
=======

- pRT3 and ``SpectralModel``: `Doriann Blain <doriann.blain@gmail.com>`_
- pRT's ``retrieval`` module: `Evert Nasedkin <nasedkinevert@gmail.com>`_

For general inquiries about pRT, contributing, etc., please contact Paul Mollière.
