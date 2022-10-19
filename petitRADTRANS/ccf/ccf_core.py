"""Core cross-correlation-related functions.
"""

import numpy as np
from scipy.interpolate import interp1d


def co_add_cross_correlation(cross_correlation, velocities_ccf, co_added_velocities):
    n_integrations = np.shape(cross_correlation)[0]

    if np.ndim(co_added_velocities) != 3:
        raise ValueError(f"rest velocities must have 3 dimensions, but has {np.ndim(co_added_velocities)}")
    elif np.shape(co_added_velocities)[1] != n_integrations:
        raise ValueError(f"size of rest velocities dimension 1 must be the size of cross correlation dimension 0 "
                         f"({n_integrations}), but is of shape {np.shape(co_added_velocities)}")

    co_added_ccf = np.zeros((np.shape(co_added_velocities)[0], np.shape(co_added_velocities)[2]))  # (Kp, Vr)

    # Interpolate the CCF at a given integration over the new (Kp, Vr) space
    ccf_interpolation_fs = [interp1d(velocities_ccf, ccf) for ccf in cross_correlation]

    for i, v_rest in enumerate(co_added_velocities):
        for j in range(n_integrations):
            co_added_ccf[i, :] += ccf_interpolation_fs[j](v_rest[j, :])  # sum every integration at a given Kp

    return co_added_ccf


def cross_correlate(array_1, array_2):
    array_1_2 = array_1 @ array_1
    array_2_2 = array_2 @ array_2
    array_mul = array_1 @ array_2

    return array_mul / np.sqrt(array_1_2 * array_2_2)


def cross_correlate_3d(matrix_1, matrix_2):
    matrix_mul = matrix_1 * matrix_2

    # Ignore indices where both matrices are NaNs: any number times NaN will return NaN, hence matrix_mul can be used
    ids = np.isnan(matrix_mul)

    matrix_1[ids] = 0  # zeros won't add to the sum
    matrix_2[ids] = 0
    matrix_mul[ids] = 0

    # Intermediate calculations
    matrix_1 = matrix_1 ** 2
    matrix_2 = matrix_2 ** 2
    matrix_1 = np.sum(matrix_1, axis=-1)
    matrix_2 = np.sum(matrix_2, axis=-1)

    matrix_mul = np.sum(matrix_mul, axis=-1)

    # Use matrix 1 to build the divisor in order to save RAM
    matrix_1 *= matrix_2
    matrix_1 = np.sqrt(matrix_1)

    # Use matrix 2 to store the result in order to save RAM
    matrix_2.fill(0.0)

    ids = np.nonzero(matrix_1)  # prevent division by 0

    # Cross correlation
    matrix_2[ids] = matrix_mul[ids] / matrix_1[ids]

    return matrix_2

