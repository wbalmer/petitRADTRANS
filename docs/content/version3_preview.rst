petitRADTRANS 3: most notable changes
=====================================

Preface
-------

petitRADTRANS 3 comes with a lot of quality-of-life features,
optimization, and code rationalization, but also with breaking
changes. A list of those breaking changes is provided
[here](https://docs.google.com/spreadsheets/d/1yCiyPJfUXzsd9gWTt3HdwntNM2MrHNfaZoacXkg1KLk/edit#gid=2092634402).
Hereafter is a short summary of the most notables changes.

Added
-----

-  Full integration of ``SpectralModel``: a convenient and modular way
   to manage your models and run retrievals, both for low resolution and
   high resolution observations.
-  Full integration of ``Planet``: the latest NASA exoplanet archive
   data accessible as easily as ``Planet.get("<planet name>")``!
-  Automatic opacity files download: missing a file? Let petitRADTRANS
   download it for you! As a plus, now downloading the 12 GB default
   input_data folder at installation is no longer necessary!
-  Support for HDF5 opacity files: load opacities faster than ever!
-  Helpful error and warning messages.
-  Possibility to retrieve or optimize uncertainties (use with caution).
-  Data preparation of high-resolution observations:
   remove telluric contaminations with SysRem or
   polynomial fitting.
-  Simple transit light loss modelling (ingress, egress) for ``SpectralModel``.
-  Useful built-in functions: convert from Bayes factor to sigma
   significance, calculate uncertainties, orbital phases, and more, with
   easy-to-use functions.

Changed
-------

-  Functions, arguments and attributes now have clearer names.
-  Spectral functions of ``Radtrans`` (``calculate_flux`` and
   ``calculate_transit_radii``) now return wavelengths, spectrum, and a
   dictionary containing additional outputs, instead of nothing.
-  Function ``Radtrans.calculate_flux`` now output by default
   wavelengths in cm (instead of frequencies in Hz) and flux in
   erg.s-1.cm-2/cm instead of erg.s-1.cm-2/Hz. Setting the argument
   ``frequencies_to_wavelengths=False`` restores the previous behaviour.
-  Function ``Radtrans.calculate_transit_radii`` now output by default
   wavelengths in cm (instead of frequencies in Hz). Setting the
   argument ``frequencies_to_wavelengths=False`` restores the previous
   behaviour.
-  Object ``Radtrans`` is now imported using
   ``from petitRADTRANS.radtrans import Radtrans`` (was
   ``from petitRADTRANS import Radtrans``) for more stable installation.
-  Improved petitRADTRANS memory usage.
-  Input data path is now stored in a config file within the folder
   <HOME>/.petitRADTRANS, generated when installing the package or using
   it for the first time.

Removed
-------

-  Multiple ``Radtrans`` attributes, some are now function outputs.
