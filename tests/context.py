import os
import sys

# Ensure that we are testing the package installed (*not* the development) files
del sys.path[0]  # remove current directory from searching path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add the package directory at the end

import petitRADTRANS

# Fortran extensions
import petitRADTRANS.fortran_chemistry
import petitRADTRANS.fortran_rebin
import petitRADTRANS.fortran_inputs
import petitRADTRANS.fortran_radtrans_core
import petitRADTRANS.fortran_rebin

# Python modules
import petitRADTRANS.ccf
import petitRADTRANS.ccf.ccf
import petitRADTRANS.config
import petitRADTRANS.planet
import petitRADTRANS.spectral_model
import petitRADTRANS.stellar_spectra.phoenix
import petitRADTRANS.physical_constants
import petitRADTRANS.physics
import petitRADTRANS.chemistry
import petitRADTRANS.chemistry.pre_calculated_chemistry
import petitRADTRANS.radtrans
import petitRADTRANS.retrieval
import petitRADTRANS.retrieval.preparing
import petitRADTRANS.utils
import petitRADTRANS.math
