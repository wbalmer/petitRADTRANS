Installation
============

Prerequisites for basic installation
____________________________________

To install petitRADTRANS, without retrievals, you need to install:
    - Python 3.9+,
    - a fortran compiler, for example ``gfortran``.

**Retrievals:** further installation instructions are displayed in the :ref:`the next section<retrievalsSection>`.

Linux
`````
On Linux, install Python and the fortran compiler with:

.. code-block:: bash

    sudo apt-get install python python-pip gfortran

On some distributions, ``python`` may need to be replaced with ``python3``.

Note that a general Python recommendation is to use a `Python virtual environment <https://docs.python.org/3/library/venv.html>`_, to prevent potential conflicts.

Windows
```````
**Using WSL:** before starting a pure Windows installation, consider using the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ (WSL). This is highly recommended in order to make the most out of pRT on Windows. Follow the WSL installation instructions from the previous link, then install pRT from the WSL terminal, following the same steps as in the Linux case. **It is also highly recommended to put the "input_data" folder on the WSL side** (see below) to get the fastest performances during retrievals.

On Windows, a fortran compiler can be acquired through `MSYS2 <https://www.msys2.org/>`_ or `Visual Studio <https://visualstudio.microsoft.com/>`_.

A Python installer is available on the `Python website <https://www.python.org/>`_.

pRT can be installed both on the Windows and WSL sides. Files on WSL can be accessed from the Windows side using the path ``\\wsl.localhost\``, and files on Windows can be accessed from the WSL side using ``/mnt`` (e.g., to get into "C:\\Users" from WSL: ``cd /mnt/c/Users``). Note however than accessing files across sides is `slow <https://learn.microsoft.com/en-us/windows/wsl/setup/environment#file-storage>`_.

Mac Os
``````
On Mac OS, be sure to have `homebrew <https://brew.sh/>`_ installed.

To ensure a safe installation, execute first:

.. code-block:: bash

    brew update
    brew upgrade
    brew doctor

A list of suggestions and fixes may be displayed when executing `brew doctor`. It is highly recommended to go through all of them before proceeding.

**Important note:** ``brew install`` is highly recommended to install all the dependencies, as this minimizes the risk of conflicts and issues.

Then, install a fortran compiler with:

.. code-block:: bash

    brew install gcc

.. _retrievalsSection:

Prerequisite for retrievals: Multinest
______________________________________

If you want to use pRT's retrieval package, you need to install Multinest.
This is because for retrievals pRT uses the PyMultiNest package,
which is a Python wrapper of the nested sampling code called MultiNest.
To install Multinest, please follow the instructions provided on the
`PyMultiNest website <https://johannesbuchner.github.io/PyMultiNest/install.html#building-the-libraries>`_.

**Important note for Windows:** `MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ retrievals, that are used by default in pRT, will not work as is on Windows. This is because MultiNest requires the LAPACK and OpenMPI libraries to function. Installing LAPACK on Windows can be a `tedious process <https://icl.utk.edu/lapack-for-windows/lapack/>`_, and OpenMPI support on Windows `has been discontinued <https://www.open-mpi.org/software/ompi/v1.6/ms-windows.php>`_, meaning that it is not possible to run MultiNest retrievals in parallel, increasing significantly computation times. This can be overcome by using WSL (see installation instructions above).

After installation, link the resulting library files in order to allow PyMultiNest to find them.
This can be done by including the ``multinest/lib/`` to your ``LD_LIBRARY_PATH``.

Add this line at the end of your environment setup file ".bash_profile", ".bashrc", or ".zshrc" (depending on your operating system and shell type):

.. code-block:: bash

    LD_LIBRARY_PATH=/path/to/multinest/lib:$LD_LIBRARY_PATH

Mac+Anaconda known issue
````````````````````````
The above may not work on a Mac when using ``anaconda``.
In that case you may also need to copy the ``multinest/lib/*`` files generated during the installation
into the ``lib`` folder that your Python binary sees. This folder should be called something like ``/opt/miniconda3/envs/name_of_your conda_environment/lib/``.
You may also need the conda version of the ``mpi4py`` package, which must be installed with:

.. code-block:: bash

    conda install mpi4py

In case of troubles, executing ``brew upgrade``, ``brew update``, then following the instructions of ``brew doctor`` may help.

Pre-installation packages
_________________________
Before starting the installation of pRT, make sure to install the following Python packages with:

.. code-block:: bash

    pip install numpy meson-python ninja

On some distributions, ``pip`` may need to be replaced with ``pip3``.


Installation of petitRADTRANS via pip install
_____________________________________________
To install pRT via pip install just execute:

.. code-block:: bash

    pip install petitRADTRANS --no-build-isolation

in a terminal. Be sure to add the ``--no-build-isolation`` flag.

To be able to use the retrieval module, execute:

.. code-block:: bash

    pip install petitRADTRANS[retrievals] --no-build-isolation

Compiling pRT from source
_________________________

Download petitRADTRANS from `Gitlab <https://gitlab.com/mauricemolli/petitRADTRANS.git>`_, or clone it from GitLab via

.. code-block:: bash

    git clone https://gitlab.com/mauricemolli/petitRADTRANS.git

- In the terminal, enter the petitRADTRANS folder.
- Execute the following command in the terminal:

.. code-block:: bash

    pip install . --no-build-isolation

Be sure to add the ``--no-build-isolation`` flag.

To be able to use the retrieval module, execute:

.. code-block:: bash

    pip install .[retrievals] --no-build-isolation

The input_data folder
_____________________

pRT relies on data (opacities, stellar spectra, planet data, pre-calculated chemical abundances) to perform its calculations.
Those data will be downloaded automatically as needed. By default, the files are downloaded into the `<home>/petitRADTRANS/input_data` directory, where `<home>` is your home folder (shortcut `~` in most OS).
This can be changed by modifying the pRT config file (see getting started section).

Alternatively, the data can be accessed and downloaded `via Keeper here <https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0>`_. The planet data are fetched from the `Nasa Exoplanet Archive <https://exoplanetarchive.ipac.caltech.edu/>`_.

Testing the installation
________________________

Open a new terminal window. Then open python and type:

.. code-block:: python
		
    from petitRADTRANS.radtrans import Radtrans
    radtrans = Radtrans(line_species=['CH4'])

If you have not already manually downloaded the CH4 correlated-k opacities, this should trigger the download of the opacity file.

The last lines of the output should be:

.. code-block:: bash

    Loading Radtrans opacities...
     Loading line opacities of species 'CH4' from file '/path/to/input_data/opacities/lines/correlated_k/CH4/12C-1H4/12C-1H4__YT34to10.R1000_0.3-50mu.ktable.petitRADTRANS.h5'... Done.
     Successfully loaded all line opacities
    Successfully loaded all opacities

The warning about the pressure can be ignored.
