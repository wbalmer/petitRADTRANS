=====================================
petitRADTRANS 3: most notable changes
=====================================

Preface
=======
petitRADTRANS 3 comes with a lot of quality-of-life features, optimization, code rationalization, more standard opacities, but also with breaking changes. For more details, see the `detailed changelog <../_static/Radtrans_v3.0.0_detailed_changelog.html>`_.

Hereafter is a short summary of the most notables changes.

Instructions to convert pRT2 opacities in the pRT3 format are given in the `conversion section <#converting-your-custom-prt2-opacities-to-prt3-format>`_.

Added
=====
-  Full integration of ``SpectralModel``: a convenient and modular way to manage your models and run retrievals, both for low resolution and high resolution observations.
-  Full integration of ``Planet``: the latest NASA exoplanet archive data accessible as easily as ``Planet.get("<planet name>")``!
-  Automatic opacity files download: missing a file? Let petitRADTRANS download it for you! As a plus, now downloading the 12 GB default input_data folder at installation is no longer necessary!
-  Support for HDF5 opacity files: load opacities faster than ever!
- More CIA and cloud opacities, see `Available opacities species <available_opacities.html>`_.
-  Helpful error and warning messages for the most common issues.
-  Possibility to retrieve or optimize uncertainties (use with caution).
-  Data preparation of high-resolution observations: remove telluric contaminations with SysRem or polynomial fitting.
-  Simple transit light loss modelling (ingress, egress) for ``SpectralModel``.
-  Useful built-in functions: convert from Bayes factor to sigma significance, calculate uncertainties, orbital phases, and more, with easy-to-use functions.

Changed
=======
-  Functions, arguments and attributes now have clearer names (e.g., ``calc_flux()`` was changed to ``calculate_flux()``).
-  Spectral functions of ``Radtrans`` (``calculate_flux`` and ``calculate_transit_radii``) now return wavelengths, spectrum, and a dictionary containing additional outputs, instead of nothing.
-  Function ``Radtrans.calculate_flux`` now output by default wavelengths in cm (instead of frequencies in Hz) and flux in erg.s-1.cm-2/cm instead of erg.s-1.cm-2/Hz. Setting the argument ``frequencies_to_wavelengths=False`` restores the previous behaviour.
-  Function ``Radtrans.calculate_transit_radii`` now output by default wavelengths in cm (instead of frequencies in Hz). Setting the argument ``frequencies_to_wavelengths=False`` restores the previous behaviour.
-  Object ``Radtrans`` is now imported using ``from petitRADTRANS.radtrans import Radtrans`` (was ``from petitRADTRANS import Radtrans``) for more stable installation.
-  Improved petitRADTRANS memory usage and performances.
-  Input data path is now stored in a config file within the folder <HOME>/.petitRADTRANS, generated when installing the package or using it for the first time.

Removed
=======
-  Multiple ``Radtrans`` attributes, some are now function outputs.


Converting pRT2 opacities to pRT3 format
========================================

.. note:: The conversion is necessary because we switched from Fortran binary tables to `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_. Correlated-k opacity files you downloaded from ExoMol do not have to be converted, since we adopted their format. You still need to place them in differently called folders, however, see the instructions on "Importing opacity tables from the ExoMol website" `here <adding_opacities.html#importing-opacity-tables-from-the-exomol-website>`_.

- If you ever added custom opacities to pRT before pRT3 was released (May 2024) you need to convert them to pRT3 (HDF5) format before you can use them, this is explained below.
- If you have a pRT2 ``input_data`` folder full of standard pRT2 opacities (those that we supplied through Keeper) you can convert those all in one go, this is also explained below. **Note that this will only work for pRT's standard opacities, not for species you added yourself.**

.. important:: **We recommend starting with the conversion of your custom opacity species**, because you need an ``input_data`` folder in the old pRT2 format for doing this (i.e., containing the ``opa_input_files`` directory). If you run ``convert_all(clean=True)``, old pRT2 opacity files will be removed after conversion, including the ``input_data/opa_input_files`` directory.

.. caution:: If you convert all standard pRT2 opacities with ``convert_all()`` (see below) the ``input_data`` structure will be modified and can no longer be used with pRT2. If you still want to use pRT2, it is better to make a copy of the ``input_data`` directory before conversion.

Below are examples for how to do the conversion.

Correlated-k opacities
----------------------

To manually convert correlated-k opacities, you can use the following example:

.. code-block:: python

    from petitRADTRANS.__file_conversion import _correlated_k_opacities_dat2h5_external_species
    from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass

    _correlated_k_opacities_dat2h5_external_species(
        path_to_species_opacity_folder='/Users/molliere/pRTv2/input_data/opacities/lines/corr_k/CH4_hargreaves',
        path_prt2_input_data='/Users/molliere/pRTv2/input_data',
        longname='12C-1H4__HITEMP.R1000_0.1-250mu',  # see c-k file naming convention
        doi='10.3847/1538-4365/ab7a1a',
        contributor='Your name',
        description="Using HITRAN's air broadening prescription.",
        molmass=get_species_molar_mass('CH4')
    )

.. caution:: Argument ``longname`` must be a valid pRT file name, otherwise it will be rejected. The correlated-k file naming is available in the :doc:`available opacities section <available_opacities>`.

The function needs to following input parameters:

- ``path_to_species_opacity_folder``: string that gives the absolute path of the folder that contains the correlated-k opacity files in the old pRT2 format (in the example above we are converting ``'CH4_hargreaves'``.
- ``path_prt2_input_data``: absolute path of the pRT2 input data folder.
- ``longname``: The species (unique) longname following the pRT3/Exomol format, which will also be the name of the HDF5 file (leave out the ``'.h5'`` extension).
- ``doi``: DOI of the reference that describes the line list (``'10.3847/1538-4365/ab7a1a'`` points to `Hargreaves et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020ApJS..247...55H/abstract>`_ in our case). Can be left empty for internal use.
- ``contributor``: in case you want to share your HDF5 file with us (please :) ), this is the contributor name we will mention in the `available opacities section <available_opacities.html>`_.
- ``description``: any additional information you think is useful to know for a user.
- ``molmass``: the mass of the absorber in atomic mass units.

After conversion the new HDF5 file will be placed into your pRT2 input data folder, in the above example in ``'/Users/molliere/pRTv2/input_data/opacities/lines/corr_k/'``. You then need to move the file ``12C-1H4__HITEMP.R1000_0.1-250mu.ktable.petitRADTRANS.h5`` from there into the pRT3 folder, following the folder structure `described for adding Exomol opacities <adding_opacities.html#importing-opacity-tables-from-the-exomol-website>`_. In our example here, the new path of the file is is ``/Users/molliere/pRT3/input_data/opacities/lines/correlated_k/CH4/12C-1H4/``. Note the change in the path to the input folder of pRT3. Also do not forget to adapt your absolute paths accordingly (very likely you do not have a folder called ``molliere``, for example).

Line-by-line opacities
----------------------

To manually convert line-by-line opacities, you can proceed as follow:

First, move the folder containing your pRT2-formatted opacities of the species you want to convert to the pRT3 input data folder, using the folder structure `described for adding Exomol opacities <adding_opacities.html#importing-opacity-tables-from-the-exomol-website>`_. For :math:`\rm CH_4`'s main isotopologue, this would correspond to ``/path/to/input_data/opacities/lines/line_by_line/CH4/12C-1H4/pRT2_CH4_directory`` (here ``pRT2_CH4_directory`` is the directory you moved, you don't need to change its name). Then, execute the following in a Python console:

.. code-block:: python

    from petitRADTRANS.__file_conversion import line_by_line_opacities_dat2h5
    from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass

    line_by_line_opacities_dat2h5(
        directory='/path/to/input_data/opacities/lines/line_by_line/species/isotopologue/old_directory',  # change accordingly
        path_input_data='/path/to/old/pRT2/input_data', # path to old pRT input data folder
        output_name='pRT_valid_opacity_filename',  # e.g., '12C-1H4__HITEMP.R1e6_0.3-28mu', see lbl file naming convention
        molmass=get_species_molar_mass('SpeciesChemicalFormula'),  # change accordingly (e.g. '12C-1H4')
        doi='doi of the opacity source',  # change accordingly, can be left empty for personal use
        contributor='Your name',  # change accordingly, can be left empty for personal use
        clean=True  # if True, automatically remove the old pRT2 opacity files stored in "directory"
    )

.. caution:: Argument ``output_name`` must be a valid pRT file name, otherwise it will be rejected. The line-by-line file naming convention is available in the :doc:`available opacities section <available_opacities>`.

If you have put your old directory at the correct place, the resulting file should already be in the correct position (here, ``'/path/to/input_data/opacities/lines/line_by_line/species/isotopologue/pRT_valid_opacity_filename.xsec.petitRADTRANS.h5'``).

Automatic conversion of the pRT2 input_data folder
--------------------------------------------------

Once you have set the path to your input_data folder (see `"Getting started" <content/notebooks/getting_started.html>`_) The simplest way to convert you pRT2 opacities into the pRT3 format is to use the provided ``convert_all`` function:

.. code-block:: python

    from petitRADTRANS.__file_conversion import convert_all

    convert_all(clean=True)  # to not remove the old files automatically, set clean to False

.. important:: If you want to keep the pRT2-formatted files, you should use ``clean=False``. Note that some of these files will be displaced, and hence will **no longer be usable as is by pRT2**. Running ``clean=True`` will minimize the impact of the conversion on your storage.

Note that this will only convert the pRT2 **default** opacities. Custom-made opacities need to be converted manually (see above sections).
