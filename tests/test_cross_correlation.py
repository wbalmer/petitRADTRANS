"""Test petitRADTRANS core CCF module.
"""
import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import reference_filenames

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


def test_1d_cross_correlation():
    reference_data = np.load(reference_filenames['simple_spectrum'])

    benchmark = Benchmark(
        function=petitRADTRANS.ccf.ccf_core.cross_correlate_matrices,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        matrix_1=reference_data['spectrum'],
        matrix_2=reference_data['spectrum']
    )
