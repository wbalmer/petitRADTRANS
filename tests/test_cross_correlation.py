"""Test petitRADTRANS core CCF module.
"""
import numpy as np
from .context import petitRADTRANS

from .utils import compare_from_reference_file, reference_filenames

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


def test_1d_cross_correlation():
    reference_data = np.load(reference_filenames['simple_spectrum'])

    cross_correlation = petitRADTRANS.ccf.ccf_core.cross_correlate_matrices(
        matrix_1=reference_data['spectrum'],
        matrix_2=reference_data['spectrum']
    )  # correlate a simple 1D spectrum with itself

    compare_from_reference_file(
        reference_filenames['simple_spectrum'],
        comparison_dict={
            'cross_correlation': cross_correlation
        },
        relative_tolerance=relative_tolerance
    )
