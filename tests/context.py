import os
import sys

# Ensure that we are testing the package installed (*not* the development) files
del sys.path[0]  # remove current directory from searching path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add the package directory at the end

# Fortran extensions

# Python modules
