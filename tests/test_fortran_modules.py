"""Test the fortran modules.
"""
import numpy as np

from .context import petitRADTRANS
from .benchmark import Benchmark
from .utils import reference_filenames

relative_tolerance = 1e-6  # relative tolerance when comparing with older results


def test_rebin_spectrum():
    reference_data = np.load(reference_filenames['simple_spectrum'])

    benchmark = Benchmark(
        function=petitRADTRANS.fortran_rebin.fortran_rebin.rebin_spectrum,
        relative_tolerance=relative_tolerance
    )

    benchmark.run(
        input_wavelengths=reference_data['wavelengths'],
        input_spectrum=reference_data['spectrum'],
        rebinned_wavelengths=reference_data['rebin_wavelengths']
    )
