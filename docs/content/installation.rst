Installation
============

Before installation: download the opacity data
______________________________________________

Before you install pRT, please download the opacity data, at least the
low-resolution version (:math:`\lambda/\Delta\lambda=1000`), as it
provides all relevant input files for pRT to run, and contains the
necessary folder structure if you want to install high-resolution
opacities later (:math:`\lambda/\Delta\lambda=10^6`).

Thus, to get started download the `opacity and input data
<https://keeper.mpdl.mpg.de/f/78b3c66857924b5aacdd/?dl=1>`_
(12.1 GB), unzip them, and put the "input_data" folder somewhere on
your computer (it does not matter where).

Next, please add the following environment variable to your
“.bash_profile”, “.bashrc”, or ".zshrc" file (depending on your operating system and shell type)
by typing 

.. code-block:: bash

   echo 'export pRT_input_data_path="absolute/path/of/the/folder/input_data"' >>~/.bash_profile

for Mac OS and

.. code-block:: bash

   echo 'export pRT_input_data_path="absolute/path/of/the/folder/input_data"' >>~/.bashrc

for Linux. Now you are ready to go and can proceed with the actual
installation of pRT.

.. attention::
   Don’t forget to adapt the path in the line above! If you are
   uncertain what the absolute path of the input_data folder is, then
   switch to that folder in the terminal, type “pwd”, and press Enter.
   You can then just copy-paste that path. Then close and reopen the
   terminal such that it will read the environment variable correctly.

If you want to also use high-resolution opacity
data please follow these steps here, but note that they can be
installed at any point after the pRT installation:

The high resolution (:math:`\lambda/\Delta\lambda=10^6`) opacity data
(about 240 GB if you want to get all species) can be
accessed and downloaded `via Keeper here`_. To
install them, create a folder called "line_by_line" in the
"input_data/opacities/lines" folder. Then put the folder of the absorber
species you downloaded in there.

.. _`via Keeper here`: https://keeper.mpdl.mpg.de/d/e627411309ba4597a343/

If you want to run retrievals: install Multinest
________________________________________________

If you want to use pRT's retrieval package, you need to install Multinest.
This is because for retrievals pRT uses the PyMultiNest package,
which is a Python wrapper of the nested sampling code called MultiNest.
To install Multinest, please follow the instructions provided on the
`PyMultiNest website <https://johannesbuchner.github.io/PyMultiNest/install.html#building-the-libraries>`_.

After installation, it is important to copy the resulting library files to a location where PyMultiNest can find them.
In that case you also need to copy the ``multinest/lib/*`` files generated during the installation
into the ``lib`` folder that your Python binary sees.
If you use anaconda, this folder should be called something like ``/opt/miniconda3/envs/name_of_your conda_environment/lib/``,
at least on a Mac. The solution suggested on the PyMultiNest website ("Include the lib/ directory in your ``LD_LIBRARY_PATH``")
does not appear to work, at least not on a Mac.

Installation of petitRADTRANS via pip install
_____________________________________________
pRT version 2.x requires a python version between 3.8 and 3.11 inclusive (version 3.12 compatibility will be added in version 3.0).
Make sure you have numpy and a fortran compiler (e.g., gfortran) installed. Then, to install pRT via pip install just type

.. code-block:: bash

   pip install petitRADTRANS

in a terminal. Note that you must also have downloaded the low-resolution
opacities either before or after to actually run pRT, see
`above <#pre-installation-download-the-opacity-data>`_.

Compiling pRT from source
_________________________

Download petitRADTRANS from `Gitlab <https://gitlab.com/mauricemolli/petitRADTRANS.git>`_, or clone it from GitLab via

.. code-block:: bash
		
   git clone https://gitlab.com/mauricemolli/petitRADTRANS.git

- In the terminal, enter the petitRADTRANS folder
- Before continuing to the next step, make sure you have numpy and a fortran compiler (e.g., gfortran) installed.
- Execute the following command in the terminal: ``pip install .``

Windows 10 and 11 instructions
_____________________

The installation of pRT on Windows machines, just as in the Linux/Mac case, requires C and Fortran compilers. Those can be obtained from, for example, `MSYS2 <https://www.msys2.org/>`_ or `Visual Studio <https://visualstudio.microsoft.com/>`_. The installation process is otherwise the same as in Linux.

**Important note:** `MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ retrievals, that are used by default in pRT, will not work as is on Windows. This is because MultiNest requires the LAPACK and OpenMPI libraries to function. Installing LAPACK on Windows can be a `tedious process <https://icl.utk.edu/lapack-for-windows/lapack/>`_, and OpenMPI support on Windows `has been discontinued <https://www.open-mpi.org/software/ompi/v1.6/ms-windows.php>`_, meaning that it is not possible to run MultiNest retrievals in parallel, increasing significantly computation times. This can be overcome by using WSL (see below).

**Using WSL:** it is highly recommended to use the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ (WSL) in order to make the most out of pRT on Windows. Follow the WSL installation instructions from the previous link, then install pRT from the WSL terminal, following the same steps as in the Linux case. **It is also highly recommended to put the "input_data" folder on the WSL side** to get the fastest performances during retrievals.

pRT can be installed both on the Windows and WSL sides. Files on WSL can be accessed from the Windows side using the path ``\\wsl$\``, and files on Windows can be accessed from the WSL side using ``/mnt`` (e.g., to get into "C:\\Users" from WSL: ``cd /mnt/c/Users``). Note however than accessing files across sides is `slow <https://learn.microsoft.com/en-us/windows/wsl/setup/environment#file-storage>`_.

Instructions for Apple silicon (M1/M2/M3)
_________________________________________

petitRADTRANS should natively install on Apple silicon machines (so M1, M2 or M3 chips).
Just make sure you have numpy, Apple's command line tools and
the `Apple silicon version of gfortran <https://github.com/fxcoudert/gfortran-for-macOS/releases>`_ installed.

Testing the installation
________________________

Open a new terminal window (this will source the ``pRT_input_data_path``). Then open python and type

.. code-block:: python
		
   from petitRADTRANS import Radtrans
   atmosphere = Radtrans(line_species = ['CH4'])

This should produce the following output:

.. code-block:: bash
		
     Read line opacities of CH4...
    Done.


Common issues
_____________

It may happen that after installation you get the following error message when trying to import pRT:

.. code-block:: bash

    ImportError: cannot import name 'fort_input' from partially initialized module 'petitRADTRANS' (most likely due to a circular import)

This usually occurs if there are multiple (conflicting) Python installations. In this case, we recommend
installing pRT in a new (clean) Python environment (e.g., using conda).
