"""Test the fortran modules.
"""
import numpy as np

from .context import petitRADTRANS
from .utils import compare_from_reference_file, reference_filenames

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


def test_rebin_spectrum():
    reference_data = np.load(reference_filenames['simple_spectrum'])

    rebin_spectrum = petitRADTRANS.fortran_rebin.fortran_rebin.rebin_spectrum(
            reference_data['wavelengths'], reference_data['spectrum'], reference_data['rebin_wavelengths']
    )

    compare_from_reference_file(
        reference_filenames['simple_spectrum'],
        comparison_dict={
            'rebin_spectrum': rebin_spectrum
        },
        relative_tolerance=relative_tolerance
    )
