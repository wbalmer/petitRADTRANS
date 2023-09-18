import os
import sys

# Ensure that we are testing the package installed (*not* the development) files
del sys.path[0]  # remove current directory from searching path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add the package directory at the end

import petitRADTRANS

# Fortran extensions
import petitRADTRANS.chem_fortran_util
import petitRADTRANS.fortran_rebin
import petitRADTRANS.fortran_inputs
import petitRADTRANS.fortran_radtrans_core
import petitRADTRANS.rebin_give_width

# Python modules
import petitRADTRANS.ccf
import petitRADTRANS.ccf.ccf
import petitRADTRANS.containers
import petitRADTRANS.containers.planet
import petitRADTRANS.containers.spectral_model
import petitRADTRANS.phoenix
import petitRADTRANS.physical_constants
import petitRADTRANS.physics
import petitRADTRANS.poor_mans_nonequ_chem
import petitRADTRANS.poor_mans_nonequ_chem.poor_mans_nonequ_chem
import petitRADTRANS.radtrans
import petitRADTRANS.retrieval
