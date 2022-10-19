"""Stores useful generic functions.
"""
import copy

import numpy as np


def box_car_conv(array, points):
    res = np.zeros_like(array)
    len_arr = len(array)

    for i in range(len(array)):
        if (i - points / 2 >= 0) and (i + points / 2 <= len_arr + 1):
            smooth_val = array[i - points / 2:i + points / 2]
            res[i] = np.sum(smooth_val) / len(smooth_val)
        elif i + points / 2 > len_arr + 1:
            len_use = len_arr + 1 - i
            smooth_val = array[i - len_use:i + len_use]
            res[i] = np.sum(smooth_val) / len(smooth_val)
        elif i - points / 2 < 0:
            smooth_val = array[:max(2 * i, 1)]
            res[i] = np.sum(smooth_val) / len(smooth_val)
    return res


logs_g = np.array([12., 10.93])

logs_met = np.array([1.05, 1.38, 2.7, 8.43, 7.83, 8.69, 4.56, 7.93, 6.24, 7.6, 6.45, 7.51, 5.41,
                     7.12, 5.5, 6.4, 5.03, 6.34, 3.15, 4.95, 3.93, 5.64, 5.43, 7.5,
                     4.99, 6.22, 4.19, 4.56, 3.04, 3.65, 3.25, 2.52, 2.87, 2.21, 2.58,
                     1.46, 1.88])


def calc_met(f):
    return np.log10((f / (np.sum(1e1 ** logs_g) + f * np.sum(1e1 ** logs_met)))
                    / (1. / (np.sum(1e1 ** logs_g) + np.sum(1e1 ** logs_met))))


def gaussian_weights1d(sigma, truncate=4.0):
    """Compute a 1D Gaussian convolution kernel.
    To be used with scipy.ndimage.convolve1d.

    Based on scipy.ndimage gaussian_filter1d and _gaussian_kernel1d.

    Args:
        sigma:
            Standard deviation for Gaussian kernel.
        truncate:
            Truncate the filter at this many standard deviations.

    Returns:

    """
    sd = float(sigma)

    # Make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * sd + 0.5)

    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sd ** 2 * x ** 2)

    return phi_x / phi_x.sum()


def gaussian_weights_running(sigmas, truncate=4.0):
    """Compute 1D Gaussian convolution kernels for an array of standard deviations.

    Based on scipy.ndimage gaussian_filter1d and _gaussian_kernel1d.

    Args:
        sigmas:
            Standard deviations for Gaussian kernel.
        truncate:
            Truncate the filter at this many standard deviations.

    Returns:

    """
    # Make the radius of the filter equal to truncate standard deviations
    radius = int(truncate * np.max(sigmas) + 0.5)

    x = np.arange(-radius, radius + 1)
    sd = np.tile(sigmas, (x.size, 1)).T

    phi_x = np.exp(-0.5 / sd ** 2 * x ** 2)

    return np.transpose(phi_x.T / phi_x.sum(axis=1))


def read_abunds(path):
    f = open(path)
    header = f.readlines()[0][:-1]
    f.close()
    ret = {}

    dat = np.genfromtxt(path)
    ret['P'] = dat[:, 0]
    ret['T'] = dat[:, 1]
    ret['rho'] = dat[:, 2]

    for i in range(int((len(header) - 21) / 22)):
        if i % 2 == 0:
            name = header[21 + i * 22:21 + (i + 1) * 22][3:].replace(' ', '')
            number = int(header[21 + i * 22:21 + (i + 1) * 22][0:3])
            # print(name)
            ret['m' + name] = dat[:, number]
        else:
            name = header[21 + i * 22:21 + (i + 1) * 22][3:].replace(' ', '')
            number = int(header[21 + i * 22:21 + (i + 1) * 22][0:3])
            # print(name)
            ret['n' + name] = dat[:, number]

    return ret


def remove_mask(data, data_uncertainties):
    """Remove masked values of 3D data and linked uncertainties. TODO generalize this
    An array of objects is created if the resulting array is jagged.

    Args:
        data: 3D masked array
        data_uncertainties: 3D masked array

    Returns:
        The data and errors without the data masked values, and the mask of the original data array.
    """
    data_ = []
    error_ = []
    mask_ = copy.copy(data.mask)
    lengths = []

    for i in range(data.shape[0]):
        data_.append([])
        error_.append([])

        for j in range(data.shape[1]):
            data_[i].append(np.array(
                data[i, j, ~mask_[i, j, :]]
            ))
            error_[i].append(np.array(data_uncertainties[i, j, ~mask_[i, j, :]]))
            lengths.append(data_[i][j].size)

    # Handle jagged arrays
    if np.all(np.asarray(lengths) == lengths[0]):
        data_ = np.asarray(data_)
        error_ = np.asarray(error_)
    else:
        print("Array is jagged, generating object array...")
        data_ = np.asarray(data_, dtype=object)
        error_ = np.asarray(error_, dtype=object)

    return data_, error_, mask_


def running_mean(x, n):
    cum_sum = np.cumsum(np.insert(x, 0, 0))

    return (cum_sum[n:] - cum_sum[:-n]) / float(n)
