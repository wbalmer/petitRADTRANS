import importlib.metadata
import os
import sys

__version__ = importlib.metadata.version("petitRADTRANS")

# Link to the libs folders on Windows
# if sys.platform == 'win32':
#     if sys.version_info >= (3, 8):
#         '''
#         The fortran extensions need libgfortran*.dll to work but for some reason it does not (always?) work even if the
#         libgfortran*.dll directory is listed in the PATH variable environment.
#         '''
#         # Dirty but working solution: add all dirs in PATH and force Python to consider all of them as DLL dirs
#         # TODO this is probably not needed if the OS/PATH is set up correctly
#         dll_directories = os.environ['PATH'].split(";")
#
#         for dll_directory in dll_directories:
#             if dll_directory != '':
#                 os.add_dll_directory(dll_directory)
