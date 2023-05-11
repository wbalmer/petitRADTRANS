Installation
============

Pre-installation: download the opacity data
___________________________________________

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

Installation via pip install
____________________________

To install pRT via pip install just type

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
- Execute the following command in the terminal: ``pip install .``

Windows 10 and 11 instructions
_____________________

The installation of pRT on Windows machines, just as in the Linux case, requires C and Fortran compilers. Those can be obtained from, for example, `MSYS2 <https://www.msys2.org/>`_ or `Visual Studio <https://visualstudio.microsoft.com/>`_. The installation process is otherwise the same as in Linux.

**Important note:** `MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ retrievals, that are used by default in pRT, will not work as is on Windows. This is because MultiNest requires the LAPACK and OpenMPI libraries to function. Installing LAPACK on Windows can be a `tedious process <https://icl.utk.edu/lapack-for-windows/lapack/>`_, and OpenMPI support on Windows `has been discontinued <https://www.open-mpi.org/software/ompi/v1.6/ms-windows.php>`_, meaning that it is not possible to run MultiNest retrievals in parallel, increasing significantly computation times. This can be overcome by using WSL (see below).

**Using WSL:** it is highly recommended to use the `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_ (WSL) in order to make the most out of pRT on Windows. Follow the WSL installation instructions from the previous link, then install pRT from the WSL terminal, following the same steps as in the Linux case. **It is also highly recommended to put the "input_data" folder on the WSL side** to get the fastest performances during retrievals.

pRT can be installed both on the Windows and WSL sides. Files on WSL can be accessed from the Windows side using the path ``\\wsl$\``, and files on Windows can be accessed from the WSL side using ``/mnt`` (e.g., to get into "C:\\Users" from WSL: ``cd /mnt/c/Users``). Note however than accessing files across sides is `slow <https://learn.microsoft.com/en-us/windows/wsl/setup/environment#file-storage>`_.

Apple M1 instructions
_____________________

The installation of pRT on Apple machines with the M1 chip requires Intel emulation with Rosetta.

.. code-block:: bash

   softwareupdate --install-rosetta

Now go to the Applications folder and find the iTerm icon. Make a copy of this application and name the new copy as something like "iTerm_Rosetta". Right click iTerm_Rosetta, choose "Get Info", and select the "Open using Rosetta" box. To test that you are indeed using the Intel emulator, type the following in your iTerm_Rosetta:

.. code-block:: bash

   arch

This command should return ``i386``.

Next, install homebrew with Rosetta:

.. code-block:: bash

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

With the Intel emulation, Homebrew should be installed at ``/usr/local/bin/brew``. Add the following to your ``.bash_profile``

.. code-block:: bash

   alias brew_i386="/usr/local/bin/brew"

In the future, you will use `brew_i386` as an alternative of `brew` with the Intel emulation.

For completeness only, you might also install Homebrew in your M1 terminal, which should be then installed at ``/opt/homebrew/bin/brew``. Add the following to your ``.bash_profile``

.. code-block:: bash

   alias brew="/opt/homebrew/bin/brew"

Now we will install ``miniconda3`` in Rosetta, but before that, we will have to modify ``.bash_profile`` so we could handle the ``conda`` between M1 and Rosetta separately. Here I assume you already installed ``anaconda`` in your M1 terminal, so the following block should be in your ``.bash_profile``:

.. code-block:: bash

   # >>> conda initialize >>>
   # !! Contents within this block are managed by 'conda init' !!
   __conda_setup="$('/Users/xxxx/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
   if [ $? -eq 0 ]; then
      eval "$__conda_setup"
   else
      if [ -f "/Users/xxxx/anaconda3/etc/profile.d/conda.sh" ]; then
          . "/Users/xxxx/anaconda3/etc/profile.d/conda.sh"
      else
          export PATH="/Users/xxxx/anaconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup
  # <<< conda initialize <<<

Note that the "xxxx" here should be your username. Let's cut these few lines and paste them into a separate file ``.init_conda_arm64.sh`` in the home directory. We will come back to handle this file later.

Now let's install ``miniconda3`` in Rosetta. First, type the following line in iTerm_Rosetta:

.. code-block:: bash

   curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh > Miniconda3-latest-MacOSX-x86_64.sh

Then type the following and follow instructions to proceed with the installation:

.. code-block:: bash

   bash Miniconda3-latest-MacOSX-x86_64.sh

Once the installation succeed, you will see that the following several new lines have been added to ``.bash_profile``:

.. code-block:: bash

   # >>> conda initialize >>>
   # !! Contents within this block are managed by 'conda init' !!
   __conda_setup="$('/Users/xxxx/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
   if [ $? -eq 0 ]; then
       eval "$__conda_setup"
   else
       if [ -f "/Users/xxxx/miniconda3/etc/profile.d/conda.sh" ]; then
           . "/Users/xxxx/miniconda3/etc/profile.d/conda.sh"
       else
           export PATH="/Users/xxxx/miniconda3/bin:$PATH"
       fi
   fi
   unset __conda_setup
   # <<< conda initialize <<<

Let's cut these few lines again and paste them into a separate file ``.init_conda_x86_64.sh`` in the home directory. In the same iTerm_Rosetta, type the following:

.. code-block:: bash

   conda config --add channels defaults
   conda config --add channels bioconda
   conda config --add channels conda-forge

Okay, now we are ready to go ahead mofify ``.bash_profile`` to handle two versions of ``conda`` between M1 and Rosetta terminals. Add the following lines to your ``.bash_profile``:

.. code-block:: bash

   # <<<<<< Added by TR 20220405 <<
   arch_name="$(uname -m)"

   if [ "${arch_name}" = "x86_64" ]; then
       echo "Running on Rosetta using miniconda3"
       source ~/.init_conda_x86_64.sh
   elif [ "${arch_name}" = "arm64" ]; then
       echo "Running on ARM64 using anaconda"
       source ~/.init_conda_arm64.sh
   else
       echo "Unknown architecture: ${arch_name}"
   fi
   # <<<<<<<< end <<<<<<<

Now, when you open iTerm / iTerm_Rosetta, you will instantly know which ``conda`` version is being used.

Next, we should install the following packages in ``miniconda3``:

.. code-block:: bash

   conda install ipython
   conda install numpy
   conda install jupyter
   conda install -c conda-forge pymultinest

Then, we install ``gfortran`` in iTerm_Rosetta:

.. code-block:: bash

   brew_i386 install gfortran

Everything is ready now, so we should simply install pRT as follow:

.. code-block:: bash

   pip install petitRADTRANS

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
