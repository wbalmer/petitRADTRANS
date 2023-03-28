"""Stores useful generic functions.
"""
import copy
import warnings

import h5py
import numpy as np

# from petitRADTRANS.fort_rebin import fort_rebin as fr  # future
from petitRADTRANS import fort_rebin as fr


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


# TODO are these actually used outside of calc_met?
logs_g = np.array([12., 10.93])

logs_met = np.array([1.05, 1.38, 2.7, 8.43, 7.83, 8.69, 4.56, 7.93, 6.24, 7.6, 6.45, 7.51, 5.41,
                     7.12, 5.5, 6.4, 5.03, 6.34, 3.15, 4.95, 3.93, 5.64, 5.43, 7.5,
                     4.99, 6.22, 4.19, 4.56, 3.04, 3.65, 3.25, 2.52, 2.87, 2.21, 2.58,
                     1.46, 1.88])


def calc_met(f):
    return np.log10((f / (np.sum(1e1 ** logs_g) + f * np.sum(1e1 ** logs_met)))
                    / (1. / (np.sum(1e1 ** logs_g) + np.sum(1e1 ** logs_met))))


def calculate_chi2(data, model, uncertainties):
    return np.sum(((data - model) / uncertainties) ** 2)


def calculate_reduced_chi2(data, model, uncertainties, degrees_of_freedom=0):
    return calculate_chi2(data, model, uncertainties) / (np.size(data) - degrees_of_freedom)


def calculate_uncertainty(derivatives, uncertainties, covariance_matrix=None):
    """
    Calculate the uncertainty of a function f(x, y, ...) with uncertainties on x, y, ... and Pearson's correlation
    coefficients between x, y, ...
    The function must be (approximately) linear with its variables within the uncertainties of said variables.
    For independent variables, set the covariance matrix to identity.
    Uncertainties can be asymmetric, in that case for N variables, use a (N, 2) array for the uncertainties.
    Asymmetric uncertainties are handled **the wrong way** (see source 2), but it is better than nothing.

    Sources:
        1. https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        2. https://phas.ubc.ca/~oser/p509/Lec_10.pdf
        3. http://math.jacobs-university.de/oliver/teaching/jacobs/fall2015/esm106/handouts/error-propagation.pdf
    Args:
        derivatives: partial derivatives of the function with respect to each variables (df/dx, df/dy, ...)
        uncertainties: uncertainties of each variable (either a 1D-array or a 2D-array containing - and + unc.)
        covariance_matrix: covariance matrix between the variables, by default set to the identity matrix

    Returns:
        A size-2 array containing the - and + uncertainties of the function
    """
    if covariance_matrix is None:
        covariance_matrix = np.identity(np.size(derivatives))

    if np.ndim(uncertainties) == 1:
        sigmas = derivatives * uncertainties

        return np.sqrt(np.matmul(sigmas, np.matmul(covariance_matrix, np.transpose(sigmas))))
    elif np.ndim(uncertainties) == 2:
        sigma_less = derivatives * uncertainties[:, 0]
        sigma_more = derivatives * uncertainties[:, 1]

        return np.sqrt(np.array([  # beware, this is not strictly correct
            np.matmul(sigma_less, np.matmul(covariance_matrix, np.transpose(sigma_less))),
            np.matmul(sigma_more, np.matmul(covariance_matrix, np.transpose(sigma_more)))
        ]))


def class_init_args2class_args(string):
    """Convenience code-writing function to convert a series of arguments into lines of initialisation for a class.
    Useful to quickly write the __init__ function of a class from its arguments.
    Example:
        >>> s = "arg1, arg2=0.3, arg3='a'"
        >>> print(class_init_args2class_args(s))
        output:
            self.arg1 = arg1
            self.arg2 = arg2
            self.arg3 = arg3
    """
    arguments = string.split(',')
    out_string = ''

    for argument in arguments:
        arg = argument.strip().rsplit('=', 1)[0]
        out_string += f"self.{arg} = {arg}\n"

    return out_string


def class_init_args2dict(string):
    """Convenience code-writing function to convert a series of arguments into a dictionary.
    Useful to quickly write a dictionary from a long list of arguments.
    Example:
        >>> s = "arg1, arg2=0.3, arg3='a'"
        >>> print(class_init_args2class_args(s))
        output:
            {
                'arg1': ,
                'arg2': ,
                'arg3': ,
            }
    """
    arguments = string.split(',')
    out_string = '{\n'

    for argument in arguments:
        arg = argument.strip().rsplit('=', 1)[0]
        out_string += f"    '{arg}': ,\n"

    out_string += '}'

    return out_string


def class2hdf5(obj, filename=None):
    """Convert an instance of a class into a HDF5 dataset."""
    with h5py.File(filename, 'w') as f:
        dict2hdf5(
            dictionary=obj.__dict__,
            hdf5_file=f
        )


def dataset2obj(obj):
    """Convert a HDF5 dataset into a list of objects (float, int or str)."""
    if hasattr(obj, '__iter__') and not isinstance(obj, bytes):
        new_obj = []

        for o in obj:
            new_obj.append(dataset2obj(o))

        return np.array(new_obj)
    elif isinstance(obj, bytes):
        return str(obj, 'utf-8')
    else:
        return obj


def dict2hdf5(dictionary, hdf5_file, group='/'):
    """Convert a dictionary into a HDF5 dataset."""
    for key in dictionary:
        if isinstance(dictionary[key], dict):  # create a new group for the dictionary
            new_group = group + key + '/'
            dict2hdf5(dictionary[key], hdf5_file, new_group)
        elif callable(dictionary[key]):
            print(f"Skipping callable '{key}': dtype('O') has no native HDF5 equivalent")
        else:
            if dictionary[key] is None:
                data = 'None'
            elif hasattr(dictionary[key], 'dtype'):
                if dictionary[key].dtype == 'O':
                    data = flatten_object(dictionary[key])
                else:
                    data = dictionary[key]
            else:
                data = dictionary[key]

            hdf5_file.create_dataset(
                name=group + key,
                data=data
            )


def fill_object(array, value):
    """Fill a numpy object array with a value."""
    if array.dtype == 'O':
        for i, dim in enumerate(array):
            array[i] = fill_object(dim, value)
    elif array.dtype == type(value):
        array[:] = value
    else:
        array = np.ones(array.shape, dtype=type(value)) * value

    return array


def flatten_object(array):
    """Flatten a numpy object array."""
    if array.dtype == 'O':
        array = flatten_object(np.concatenate(array))
    else:
        array = np.concatenate(array)

    return array


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


def hdf52dict(hdf5_file):
    dictionary = {}

    for key in hdf5_file:
        if isinstance(hdf5_file[key], h5py.Dataset):
            dictionary[key] = dataset2obj(hdf5_file[key][()])
        elif isinstance(hdf5_file[key], h5py.Group):
            dictionary[key] = hdf52dict(hdf5_file[key])
        else:
            warnings.warn(f"Ignoring '{key}' of type '{type(hdf5_file[key])} in HDF5 file: "
                          f"hdf52dict() can only handle types 'Dataset' and 'Group'")

    return dictionary


def mean_uncertainty(uncertainties):
    """Calculate the uncertainty of the mean of an array.

    Args:
        uncertainties: individual uncertainties of the averaged array

    Returns:
        The uncertainty of the mean of the array
    """
    return np.sqrt(np.sum(uncertainties ** 2)) / np.size(uncertainties)


def median_uncertainties(uncertainties):
    """Calculate the uncertainty of the median of an array.

    Demonstration:
        uncertainty ~ standard deviation = sqrt(variance) = sqrt(V)
        V_mean / V_median = 2 * (N - 1) / (pi * N); (see source)
        => V_median = V_mean * pi * N / (2 * (N - 1))
        => uncertainty_median = uncertainty_mean * sqrt(pi * N / (2 * (N - 1)))

    Source:
        https://mathworld.wolfram.com/StatisticalMedian.html

    Args:
        uncertainties: individual uncertainties of the median of the array

    Returns:
        The uncertainty of the median of the array
    """
    return mean_uncertainty(uncertainties) \
        * np.sqrt(np.pi * np.size(uncertainties) / (2 * (np.size(uncertainties) - 1)))


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


def rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths):
    """Re-bin the spectrum using the Fortran rebin_spectrum function, and catch errors occurring there.
    The fortran rebin function raises non-blocking errors. In that case, the function outputs an array of -1.

    Args:
        input_wavelengths: wavelengths of the input spectrum
        input_spectrum: spectrum to re-bin
        rebinned_wavelengths: wavelengths to re-bin the spectrum to. Must be contained within input_wavelengths

    Returns:
        The re-binned spectrum on the re-binned wavelengths
    """
    rebinned_spectrum = fr.rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths)

    if np.all(rebinned_spectrum == -1):
        raise ValueError(f"something went wrong during re-binning (rebin.f90), check the previous messages")
    elif np.any(rebinned_spectrum < 0):
        raise ValueError(f"negative value in re-binned spectrum, this may be related to the inputs")

    return rebinned_spectrum


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
    if np.all(np.array(lengths) == lengths[0]):
        data_ = np.array(data_)
        error_ = np.array(error_)
    else:
        print("Array is jagged, generating object array...")
        data_ = np.array(data_, dtype=object)
        error_ = np.array(error_, dtype=object)

    return data_, error_, mask_


def running_mean(x, n):
    cum_sum = np.cumsum(np.insert(x, 0, 0))

    return (cum_sum[n:] - cum_sum[:-n]) / float(n)


def savez_compressed_record(file, numpy_record_array):
    """Apply numpy.savez_compressed on a record array."""
    data_dict = {key: numpy_record_array[key] for key in numpy_record_array.dtype.names}
    np.savez_compressed(file, **data_dict)
