import numpy as np
from sys import platform
import os
import threading, subprocess

def b_range(x, b):
    if x > b:
        return -np.inf
    else:
        return 0.

def a_b_range(x, a, b):
    if x < a:
        return -np.inf
    elif x > b:
        return -np.inf
    else:
        return 0.

def show(filepath): 
    """ open the output (pdf) file for the user """
    if os.name == 'mac' or platform == 'darwin': subprocess.call(('open', filepath))
    elif os.name == 'nt' or platform == 'win32': os.startfile(filepath)
    elif platform.startswith('linux') : subprocess.call(('xdg-open', filepath))
