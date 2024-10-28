============
Installation
============

Prerequisites for basic installation
====================================
To install petitRADTRANS, without retrievals, you need to install:

- Python 3.9+,
- a fortran compiler, for example ``gfortran``.
- to use the automatic data download feature, ``chromium`` [*]_.

**Running retrievals:** read the :ref:`MultiNest installation instructions<retrievalsSection>`.

Linux
-----
On Linux, install Python and the fortran compiler with:

.. code-block:: bash

    sudo apt-get install python python-pip gfortran chromium

On some distributions, ``python`` may need to be replaced with ``python3``.

.. Note:: A general Python recommendation is to use a Python virtual environment such as `venv <https://docs.python.org/3/library/venv.html>`_ (recommended) or `conda <https://docs.anaconda.com/free/anaconda/install/index.html>`_, to prevent potential conflicts.

Mac OS
------

.. important:: On Mac, it is highly recommended to use a Python virtual environment such as `venv <https://docs.python.org/3/library/venv.html>`_ (recommended) or `conda <https://docs.anaconda.com/free/anaconda/install/index.html>`_, to prevent potential conflicts.

Recommended: using homebrew
~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Mac OS, it is highly recommended to use `homebrew <https://brew.sh/>`_. Homebrew is able to manage external libraries dependencies and can help you fix broken setups. Other installation methods are more risky by making setup-related errors frequent, and difficult to identify and to fix.

To ensure a safe installation, execute first:

.. code-block:: bash

    brew update
    brew upgrade
    brew doctor

A list of suggestions and fixes may be displayed when executing `brew doctor`. It is highly recommended to go through all of them before proceeding.

Then, install a fortran compiler with:

.. code-block:: bash

    brew install gcc

.. note:: In general, ``brew install`` is highly recommended to install all the dependencies (including conda), as this minimizes the risk of conflicts and issues.

Using gfortran disk images
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: While using **homebrew is the preferred method** for installing external libraries on Mac, alternative methods exist. Use them at your own risk.

François-Xavier Coudert's `github repository <https://github.com/fxcoudert/gfortran-for-macOS>`_ provides gfortran disk images (.dmg) with which you can install gfortran like any other program for Mac, through an installation wizard. Both Apple Silicon (M1, M2, M3) and Intel chip versions are available.

Windows
-------

Recommended: using WSL
~~~~~~~~~~~~~~~~~~~~~~
To make the most out of pRT on Windows, it is recommended to use the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ (WSL).

Follow the WSL installation instructions from the previous link, then install pRT from the WSL terminal, following the same steps as in the Linux case.

.. important:: It is highly recommended to put the "input_data" folder on the WSL side (see below) to get the fastest performance during retrievals.

Native installation prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Get a fortran compiler through, for example, `MSYS2 <https://www.msys2.org/>`_ or `Visual Studio <https://visualstudio.microsoft.com/>`_.
2. If you already have a `Chromium-based browser <https://en.wikipedia.org/wiki/Chromium_(web_browser)#Browsers_based_on_Chromium>`_, you can skip this step. Otherwise, download Chromium, for example from its `official website <https://download-chromium.appspot.com/>`_.
3. Go to the `Python website <https://www.python.org/>`_, then download and execute the Python installer.

.. warning:: It is **not** possible to run parallel (fast) retrievals with a native Windows installation (see the :ref:`MultiNest section<multinest_windows>`).

WSL-native dual installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pRT can be installed both on the Windows and WSL sides. Files on WSL can be accessed from the Windows side using the path ``\\wsl.localhost\``, and files on Windows can be accessed from the WSL side using ``/mnt`` (e.g., to get into "C:\\Users" from WSL: ``cd /mnt/c/Users``). Note however than accessing files across sides is `slow <https://learn.microsoft.com/en-us/windows/wsl/setup/environment#file-storage>`_.

.. _retrievalsSection:

Prerequisite for retrievals: MultiNest
======================================
.. _multinest_windows:

.. warning:: **Windows native installation:** `MultiNest retrievals <https://github.com/JohannesBuchner/MultiNest>`_, that are used by default in pRT, will not work as is on Windows. This is because MultiNest requires the LAPACK and OpenMPI libraries to function. Installing LAPACK on Windows can be a `tedious process <https://icl.utk.edu/lapack-for-windows/lapack/>`_, and OpenMPI support on Windows `has been discontinued <https://www.open-mpi.org/software/ompi/v1.6/ms-windows.php>`_, meaning that it is not possible to run MultiNest retrievals in parallel, increasing significantly computation times. This can be overcome by using WSL (see installation instructions above).

If you want to use pRT's retrieval package, you need to install the PyMultiNest package:

1. Follow the instructions provided on the `PyMultiNest website <https://johannesbuchner.github.io/PyMultiNest/install.html#prerequisites-for-building-the-libraries>`_.
2. Link the resulting library files by including the ``MultiNest/lib/`` to your ``LD_LIBRARY_PATH``. This can be done by adding this line at the end of your environment setup file ".bash_profile", ".bashrc", or ".zshrc" (depending on your operating system and shell type):

    .. code-block:: bash

        LD_LIBRARY_PATH=/path/to/MultiNest/lib:$LD_LIBRARY_PATH

.. warning:: **Using Mac+Anaconda:** see the :ref:`troubleshooting section<mac_anaconda_issue>`.

.. _PreInstallSection:

Pre-installation packages
=========================
Before starting the installation of pRT, make sure to install the following Python packages with:

.. code-block:: bash

    pip install numpy meson-python ninja

On some distributions, ``pip`` may need to be replaced with ``pip3``.


Installation of petitRADTRANS via pip install
=============================================
To install pRT **without retrievals** via pip install, open a terminal and run:

.. code-block:: bash

    pip install petitRADTRANS --no-build-isolation

Be sure to add the ``--no-build-isolation`` flag.

To be able to use the retrieval module, execute:

.. code-block:: bash

    pip install petitRADTRANS[retrieval] --no-build-isolation

Compiling pRT from source
=========================
1. Download petitRADTRANS from `Gitlab <https://gitlab.com/mauricemolli/petitRADTRANS.git>`_, or clone it from GitLab via:

    .. code-block:: bash

        git clone https://gitlab.com/mauricemolli/petitRADTRANS.git
2. In the terminal, enter the petitRADTRANS folder.
3. **No retrievals:** execute the following command in the terminal:

    .. code-block:: bash

        pip install . --no-build-isolation
4. **With retrievals:** execute the following command in the terminal:

    .. code-block:: bash

        pip install .[retrieval] --no-build-isolation

Setting up the input_data directory
===================================
By default, pRT's input files are downloaded into the ``<home>/petitRADTRANS/input_data`` directory, where ``<home>`` is your home directory (shortcut ``~`` in most OS).
This can be changed by modifying the pRT config file. All of this is described more in the `"Getting Started" <notebooks/getting_started.html#Configuring-the-input_data-folder>`_ notebook.
**Please note that the folder that pRT stores its opacities in has to be called** ``input_data`` (but it can be placed wherever you want).

.. note::

    pRT relies on data (opacities, stellar spectra, planet data, pre-calculated chemical abundances) to perform its calculations. Those data will be downloaded automatically as needed.

    Alternatively, the data can be accessed and downloaded `via Keeper here <https://keeper.mpdl.mpg.de/d/ccf25082fda448c8a0d0>`_. The planet data are fetched from the `Nasa Exoplanet Archive <https://exoplanetarchive.ipac.caltech.edu/>`_.

Testing the installation
========================
Open a new terminal window. Then open python and type:

.. code-block:: python

    from petitRADTRANS.radtrans import Radtrans
    radtrans = Radtrans(line_species=['CH4'])

If you have not already manually downloaded the CH4 correlated-k opacities, this should trigger the download of the opacity file. You may be prompted with the following message:

.. code-block:: bash

    More than one file detected in <long/path> and no default file set for this path in petitRADTRANS' configuration
    Please select one of the files in the list below by typing the corresponding integer:

In that case, simply choose a file among the one listed under the message, type the corresponding integer, then press "Enter". More information is available in the `tutorial <notebooks/getting_started.html#Loading-opacities>`_.

The last lines of the output should be:

.. code-block:: bash

    Loading Radtrans opacities...
     Loading line opacities of species 'CH4' from file '/path/to/input_data/opacities/lines/correlated_k/CH4/12C-1H4/12C-1H4__YT34to10.R1000_0.3-50mu.ktable.petitRADTRANS.h5'... Done.
     Successfully loaded all line opacities
    Successfully loaded all opacities

The warning about the pressure can be ignored.

Troubleshooting the installation
================================

Temporary directory issue
-------------------------
When importing ``Radtrans``, you may see one of those two errors:

.. code-block:: python

    # For a pip install
    ModuleNotFoundError: No module named 'petitRADTRANS.<fortran_extension>'

    # For an editable pip install
    FileNotFoundError: [Errno 2] No such file or directory: '/a/temporary/directory/overlay/bin/ninja'

The issue is often caused by your setup installing the fortran extensions inside a temporary directory, that is then automatically removed.
Try these fixes in that order:

1. Ensure that you added the ``--no-build-isolation`` flag to the installation command. This should fix the issue in almost all cases.
2. Ensure that all the installing elements of your setup (``pip``, ``conda``, fortran compiler, etc.) are up-to-date and installed cleanly.
3. If you are on Mac, and use Homebrew, try first to execute ``brew upgrade``, ``brew update``, then to follow the instructions of ``brew doctor``, before re-trying the installation.
4. If you are on Mac, and do not use Homebrew, the error may be related with your setup. Carefully check for libraries versions, dependencies, and duplicate installations.
5. In last resort, you can add the ``--no-clean`` flag to the installation command. Beware however: this will create a temporary directory that will not be removed from your system, taking space on your disk. Each new installation with this flag will create a new temporary directory, but will **not** remove the previous one. You may need to perform manual cleaning to free space on your disk.

Minimum Python version not available
------------------------------------
In some Linux distro where the minimum Python version necessary to run petitRADTRANS is not available, it may be necessary to install a newer version from an alternative repository, e.g. the `deadsnakes repository <https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa>`_.

In that case, execute the following (replace ``python3.x`` with the desired Python version, e.g. ``python3.13``):

.. code-block:: bash

    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install pkg-config
    sudo apt-get install python3.x
    sudo apt-get install python3.x-dev

To prevent further Python-related issues with your operating system, which can be as severe as `completely breaking it <https://github.com/python/cpython/issues/102134#issuecomment-1445428402>`_, you then need to create a virtual environment (replace ``/path/to/my_venv`` with the path where you want to install your virtual environment):

.. code-block:: bash

    sudo apt-get install python3.x-venv
    python3.x -m venv /path/to/my_venv
    source /path/to/my_venv/bin/activate

Then, follow the :ref:`standard installation instructions<PreInstallSection>`. You may encounter further issues installing third-party packages like ``setuptools`` or ``mpi4py``. In that case, you can execute, e.g.:

.. code-block:: bash

    pip install -U setuptools cython

The installation should then proceed without issue.

.. _mac_anaconda_issue:

Mac+Anaconda known issue with MultiNest
---------------------------------------
Linking the MultiNest libraries the usual way may not work on a Mac when using ``anaconda``. In that case you may also need to copy the ``MultiNest/lib/*`` files generated during the installation into the ``lib`` folder that your Python binary sees. This folder should be called something like ``/opt/miniconda3/envs/name_of_your conda_environment/lib/``. You may also need the conda version of the ``mpi4py`` package, which must be installed with:

.. code-block:: bash

    conda install mpi4py

In case of troubles, if you use Homebrew, executing ``brew upgrade``, ``brew update``, then following the instructions of ``brew doctor`` may help. If you do not use Homebrew, the error may be related with your setup. Carefully check for libraries versions, dependencies, and duplicate installations.
A common error with Apple silicon when trying to run retrievals is ``AttributeError: dlsym(RTLD_DEFAULT, run): symbol not found``. This is a problem inherited from ``pymultinest``, and to solve it you should add the following lines of code before importing pRT or ``pymultinest``.

.. code-block:: bash

    import os
    os.environ["DYLB_LIBRARY_PATH"] = "/path/to/Multinest/lib"

Other issues
------------
You can take a look at the solved issues `here <https://gitlab.com/mauricemolli/petitRADTRANS/-/issues>`_. If you do not find an helpful answer there, do not hesitate to `open a new issue <contributing.html#suggestions-and-reporting-issues>`_.

.. [*] Why Chromium? To download data from Keeper, pRT uses ``selenium`` to load webpages through a browser. When the code for this interface was written, Chromium was the only mainstream browser that could be used `without any windows opening <https://en.wikipedia.org/wiki/Headless_browser>`_. Chromium is also `free and open-source <https://en.wikipedia.org/wiki/Chromium_(web_browser)>`_. In addition, it is the base of `many widespread browsers <https://en.wikipedia.org/wiki/Chromium_(web_browser)#Browsers_based_on_Chromium>`_.
