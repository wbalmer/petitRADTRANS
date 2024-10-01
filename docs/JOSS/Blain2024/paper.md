---
title: 'SpectralModel: a high-resolution framework for petitRADTRANS 3'
tags:
  - Python
  - astronomy
  - exoplanets
  - atmospheres
languages:
  - Python
  - Fortran
authors:
  - name: Doriann Blain
    orcid: 0000-0002-1957-0455
    corresponding: true
    affiliation: 1
  - name: Paul Mollière
    orcid: 0000-0003-4096-7067
    affiliation: 1
  - name: Evert Nasedkin
    orcid: 0000-0002-9792-3121
    affiliation: 1
affiliations:
 - name: Max Planck Institut für Astronomie, DE
   index: 1
date: 18 May 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
#\footnote{\url{https://petitradtrans.readthedocs.io/en/latest/content/notebooks/pRT_Retrieval_Example.html}}. 
---

# Summary
Atmospheric characterisation from spectroscopic data is a key to understand planetary formation. Two types of observations can be performed for this kind of analysis. Space-based observations (e.g., using the James Webb Space Telescope, JWST), are not impeded by the Earth's atmosphere, but are currently limited to low resolving powers ($< 3000$), which can lead to ambiguities in some species detections. Ground-based observations (e.g., using the Very Large Telescope, VLT), on the other hand, can benefit from large resolving powers ($\approx 10^{5}$), allowing for unambiguous species detection, but are impacted by telluric spectral lines.
`petitRADTRANS` (pRT) is a radiative transfer package used for computing emission or transmission spectra of planetary atmospheres [@Molliere2019].
The package has a non-negligible user base, the original article being cited in 264 refereed works at the time of writing.
pRT is already relatively easy to use on space-based, low-resolution observations. However, while the package technically has the capacity to analyse high-resolution spectra, thanks to its ability to incorporate high-resolution ($\mathcal{R} = 10^{6}$) line lists, ground-based observations analysis is a complex and challenging task. The new `SpectralModel` object provides a powerful and flexible framework that streamlines the setup necessary to model and retrieve high-resolution spectra.

# Statement of need
Calculating a spectrum using pRT's core object `Radtrans` is a two-step process in which the user first instantiates the object, giving parameters that control the loading of opacities. The second step is for the user to call one of the `Radtrans` function, giving "spectral" parameters such as the temperatures or the mass fractions of the atmosphere, that will be used in combination with the loaded opacities to generate the spectrum. 

However, these two steps are by themselves often insufficient to build a spectrum in a real-life scenario. The spectral parameters may individually rely on arbitrarily complex models requiring their own parameters, and may depend on each other. For example, getting mass fractions from equilibrium chemistry requires knowing the temperature profile, and the mean molar mass requires knowing the mass fractions (see e.g. the built-in pRT functions). Common operations such as convolving the spectrum, scaling it to stellar flux, or more specifically for high-resolution spectra, Doppler-shifting the spectrum and including the transit effect, must be done by post-processing the `Radtrans`-generated spectrum. Finally, using a retrieval requires to code a "retrieval model" including all the steps described above. This induces, especially for first-time users, a significant setup cost. The alternative is to use one of pRT's built-in models, but this lacks flexibility.

The `SpectralModel` object extends the base capabilities of the petitRADTRANS package by providing a standardized but flexible framework for spectral calculations. It has been especially designed to effectively erase the setup cost of modelling the spectral Doppler-shift, the transit effect, and of implementing the preparation step necessary for ground-based high-resolution observations analysis. `SpectralModel` is also interfaced with pRT's `retrieval` module [@Nasedkin2024], and as such is an easy-to-use tool to perform both high- and low-resolution atmospheric retrievals. Compared to other commonly used spectral modelling packages, for example [ATMOSPHERIX](https://github.com/baptklein/ATMOSPHERIX_DATA_RED) [@Klein2023], [Brewster](https://github.com/substellar/brewster) [@Burningham2021], [CHIMERA](https://github.com/mrline/CHIMERA) [@Line2013], [PSG](https://psg.gsfc.nasa.gov/) [@VILLANUEVA2018], [NEMESIS](https://github.com/nemesiscode/radtrancode) [@IRWIN2008], [PICASO](https://github.com/natashabatalha/picaso) [@Batalha2019], [PLATON](https://github.com/ideasrule/platon) [@Zhang2020], [POSEIDON](https://github.com/MartianColonist/POSEIDON) [@MacDonald2023], [TauREx](https://github.com/ucl-exoplanets/TauREx3_public) [@Al-Refaie2021], petitRADTRANS is currently, to our knowledge, the only one able to both generate time-varying high-resolution spectra and retrieve the corresponding data out-of-the-box[^1].

[^1]: ATMOSPHERIX is able to make cross-correlation analysis of high-resolution spectra, but relies on petitRADTRANS to generate its templates. HYDRA-H [@Gandhi2019] is a code able to perform high-resolution data retrievals, but is not publicly available. The other cited packages may have out-of-the-box single-time high-resolution spectral generation capabilities, but no time-varying high-resolution data retrieval framework, similarly to petitRADTRANS before the implementation of `SpectralModel`.

The combination of ease-of-use and flexibility offered by `SpectralModel` makes it a powerful tool for high-resolution (but also low-resolution) atmospheric characterisation. With the upcoming first light of a new generation of ground based telescopes, such as the Extremely Large Telescope, `SpectralModel` makes petitRADTRANS ready for the new scientific discoveries that will be unveiled in the next era of high-resolution observations.

# The `SpectralModel` object
## Main features
### Spectral parameter calculation framework

![\label{fig:flowchart}Flowchart of `SpectralModel.calculate_spectrum` function. The annotation below the model functions represents an example of execution order of these function after topological sorting, involving the temperature ($T$), the metallicity ($Z$), the time ($t$), the mass fractions (MMR), the mean molar masses (MMW), the orbital phases ($\phi$), the relative velocities ($v$), and the transit effect ($\delta$). Additional deformations ($D$) and noise ($N$) can also be included.](flowchart.pdf)

`SpectralModel` provides a framework to automatise the calculation of the spectral parameters. Each spectral parameter is linked to a function, called here "model function", which calculates its value. This feature can be extended to the parameters required for these functions, and so on. Before calculating spectra, the function's execution order is automatically determined through a topological sorting algorithm[^2] [@Kahn1962]. `SpectralModel` comes with built-in functions [@Blain2024] for all the spectral parameters, so that the object can be used "out-of-the-box". Parameters that ultimately do not depend on any function are called "model parameters", and must be given during instantiation.

[^2]: Cyclic dependencies are not supported.

In addition, `SpectralModel` provides built-in functions [@Blain2024] to scale, convolve, Doppler-shift, rebin, include planet transit effect, and prepare a spectrum after it has been calculated. Similarly to model functions, these "spectral modification functions" must be given, if used, their own model parameters during instantiation.

The spectral calculation is done within the `calculate_spectrum` function (see \autoref{fig:flowchart}). The spectral mode (emission or transmission), as well as which of the spectral modification to activate (i.e. only scaling, or both convolving and rebinning, etc.), are controlled through the function's arguments ("spectral modification parameters").

### Automatic optimal wavelength range calculation
A way to slightly reduce the high[^3] memory usage of high-resolution spectral analysis is to load exactly the wavelength range required for an analysis, instead of relying on manual inputs. This task is complicated in high-resolution retrievals due to parameters influencing the Doppler-shift (that is, the radial velocity semi-amplitude $K_p$, the rest frame velocity shift $V_\mathrm{rest}$, and the mid transit time offset $T_0$) being retrieved. `SpectralModel` comes with a class method which takes into account the (uniform) prior range of these parameters to automatically calculate the optimal wavelength range to load.

[^3]: Loading a typical pRT line-by-line opacity file between 1 and 2 $\mu$m takes 804 MB of RAM, according to `numpy.ndarray.nbytes`.

### Interface with pRT's `retrieval` module
In order to be able to perform high-resolution data retrievals, the `Retrieval` object has been extended to support spectra with up to 3 dimensions, intended to be spectral order, exposure (time), and spectral pixel (wavelength). Several improvements to the module have been implemented as well:

- The retrieved data can now be provided as arrays instead of requiring a file.
- Custom `Radtrans` (or by extension `SpectralModel`) objects can now be used for retrievals.

In addition, `SpectralModel`'s model parameters and spectral modification functions can be advantageously used to simplify the retrieval setup compared to `Radtrans`'. This removes the need for several steps:

- building the `RetrievalConfig` object, as this has been automated,
- declaring the fixed parameters, as all model parameters that are not retrieved parameters are *de facto* fixed parameters,
- writing the retrieval model function, as it is given by the `SpectralModel` itself.

Ground-based high-resolution spectra contain telluric and stellar lines that must be removed. This is usually done with a "preparing" pipeline (also called "detrending" or "pre-processing" pipeline). To this end, a new `retrieval.preparing` sub-module has been implemented, containing the "Polyfit" pipeline [@Blain2024] and the "SysRem" pipeline [@Tamuz2005]. To perform a retrieval when the data are prepared with "Polyfit", the forward model must be prepared in the same way [@Blain2024]. This forward model preparation step can be activated when calculating a spectrum with `SpectralModel`.

### Ground-based data simulation
Data ($F$) taken from ground telescopes can be expressed as $F = M_\Theta \circ D + N$ [@Blain2024], where $M_\Theta$ is an exact model with true parameters $\Theta$, $D$ ("deformation matrix") represents the combination of telluric lines, stellar lines, and instrumental deformations (pseudo-continuum, blaze function, ...), and $N$ is the noise. The operator "$\circ$" represents the element-wise product. Telluric lines, noise, and other deformations can be included in a `SpectralModel` object. A time-varying airmass can be added as model parameter to better model the telluric lines. Finally, a command-line interface (CLI) with ESO's [SKYCALC](https://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC) sky model calculator has been implemented, adapting the CLI provided on the [ESO's website](https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html).

## Workflows
Examples for these workflows are available in the pRT's documentation.

### Spectra calculation
Calculating spectra with `SpectralModel` is done in two steps:

1. Instantiation: similarly to `Radtrans`, this step is done to load the opacities, and thus requires the same parameter as a `Radtrans` instantiation. In addition, the user can provide model parameters, that will give the spectral parameters and the modification parameters. Finally, a custom `dict` can be given if the user desires to use different functions than the built-in ones.
2. Calculation: spectral calculation is done with a unique function. The spectrum type (emission or transmission), as well as modification flags (for scaling, Doppler-shifting, etc.) are given as arguments.

### Retrievals
Retrieving spectra with `SpectralModel` is done in seven steps:

1. Loading the data,
2. For high-resolution ground-based data: preparing the data,
3. Setting the retrieved parameters, this is done by filling a `dict`,
4. Setting the forward model, by instantiating a `SpectralModel` object,
5. Instantiating a `Data` object with the `SpectralModel` dedicated function,
6. Instantiating a `Retrieval` object from the previously built `Data` object(s),
7. Running the retrieval.

In addition, a new corner plot function, based on the `corner` package [@Foreman-Mackey2016], has been implemented to ease the representation of the retrieval results with this framework.
 

# The petitRADTRANS 3 update
+--------------------------------+--------------------+-------------------+--------------------+--------------------+
| Test \label{tab:performances}  | pRT 2.7.7 time (s) | pRT 3.1.0 time (s)| pRT 2.7.7 RAM (MB) | pRT 3.1.0 RAM (MB) |
|                                |                    |                   |                    |                    |
+================================+====================+===================+====================+====================+
| Opacity loading, `'c-k'`       | 3.2                | 1.0               | --                 | --                 |
+--------------------------------+--------------------+-------------------+--------------------+--------------------+
| Opacity loading, `'lbl'`       | 6.2                | 0.5               | --                 | --                 |
+--------------------------------+--------------------+-------------------+--------------------+--------------------+
| Emission, `'c-k'`              | 6.7                | 5.4               | 3135               | 1509               |
+--------------------------------+--------------------+-------------------+--------------------+--------------------+
| Emission, `'lbl'`              | 8.1                | 5.1               | 5864               | 2643               |
+--------------------------------+--------------------+-------------------+--------------------+--------------------+
| Transmission, `'c-k'`          | 1.3                | 0.7               | 992                | 758                |
+--------------------------------+--------------------+-------------------+--------------------+--------------------+
| Transmission, `'lbl'`          | 7.0                | 3.4               | 3929               | 2010               |
+================================+====================+===================+====================+====================+
| - Times are measured using the `cProfile` standard library, from the average of 7 runs.                           |
| - "RAM": peak RAM usage as reported by the `tracemalloc` standard library.                                        |
| - `'c-k'`: using correlated-k opacities (CH$_4$ and H$_2$O), from 0.3 to 28 $\mu$m.                               |
| - `'lbl'`: using line-by-line opacities (CO and H$_2$O), from 0.9 to 1.2 $\mu$m.                                  |
| - All spectra calculations are done using 100 pressure levels. Emission scattering is activated in `'c-k'` mode.  |
| - Results obtained on Debian 12.5 (WSL2), CPU: AMD Ryzen 9 3950X @ 3.50 GHz.                                      |
+================================+====================+===================+====================+====================+

Fully and seamlessly implementing `SpectralModel` into pRT required major changes and refactors to pRT's code. The changes focus on optimisations (both for speed and RAM usage) for high-resolution spectra computing, but this also impacts the correlated-k (low-resolution) part of the code (see \autoref{tab:performances}). To speed-up "input data" (opacities, pre-calculated equilibrium chemistry table, star spectra table) loading times, pRT's loading system has been overhauled and the loaded files have been converted from a mix of ASCII, Fortran unformatted and [HDF5](https://www.hdfgroup.org/solutions/hdf5/) files to HDF5-only. Opacities now also follow an extended [ExoMol database](https://www.exomol.com/) naming and structure convention. The package's installation process has been made compatible with Python $\geq$ 3.12[^4]. Finally, several quality-of-life features (e.g., missing requested opacities can be automatically downloaded from the project's [Keeper library](https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0/), or the `Planet` object) have been implemented.

[^4]: pRT 2 used the [`numpy.distutils` module](https://numpy.org/doc/stable/reference/distutils.html) to compile its Fortran extensions. This module is deprecated and is removed for Python 3.12. pRT 3 uses the [Meson build system](https://mesonbuild.com/) instead, with almost unnoticeable changes for users.

# Acknowledgements

We thank the pRT users, who greatly helped improving the package by sharing their suggestions and reporting their issues.

# References
