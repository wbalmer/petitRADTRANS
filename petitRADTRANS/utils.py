"""Stores useful generic functions.
"""
import copy
import warnings

import h5py
import numpy as np
from scipy.special import erf, erfinv, lambertw

from petitRADTRANS.fortran_rebin import fortran_rebin as frebin


class LockedDict(dict):
    """Derivative of dict with a lock.
    Can be used to ensure that no new key is added once the lock is on, to prevent errors due to key typos.
    """
    def __init__(self):
        super().__init__()
        self._locked = False

    def __copy__(self):
        """Override the copy.copy method. Necessary to allow locked LockedDict to be copied."""
        cls = self.__class__
        result = cls.__new__(cls)

        result.unlock()  # force initialization of _locked

        # First copy the keys in the new object
        for key, value in self.items():
            result[key] = value

        # Then copy the attributes to prevent the effect of the lock
        result.__dict__.update(self.__dict__)

        return result

    def __deepcopy__(self, memo):
        """Override the copy.deepcopy method. Necessary to allow locked LockedDict to be copied."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.unlock()  # force initialization of _locked

        # First copy the keys in the new object
        for key, value in self.items():
            key = copy.deepcopy(key, memo)
            value = copy.deepcopy(value, memo)
            result[key] = value

        # Then copy the attributes to prevent the effect of the lock
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))

        return result

    def __setitem__(self, key, value):
        """Prevent a key to be added if the lock is on."""
        if key not in self and self._locked:
            raise KeyError(f"'{key}' not in LockedDict (locked), unlock the LockedDict to add new keys")
        else:
            super().__setitem__(key, value)

    def lock(self):
        self._locked = True

    def unlock(self):
        self._locked = False


# TODO are these actually used outside of calc_met?
def _get_log_gs():
    return np.array([12., 10.93])


def _get_log_mets():
    return np.array([1.05, 1.38, 2.7, 8.43, 7.83, 8.69, 4.56, 7.93, 6.24, 7.6, 6.45, 7.51, 5.41,
                     7.12, 5.5, 6.4, 5.03, 6.34, 3.15, 4.95, 3.93, 5.64, 5.43, 7.5,
                     4.99, 6.22, 4.19, 4.56, 3.04, 3.65, 3.25, 2.52, 2.87, 2.21, 2.58,
                     1.46, 1.88])


def bayes_factor2sigma(bayes_factor):
    """
    Convert a Bayes factor, or "evidence", into a sigma significance. For Bayes factor higher than exp(25), the function
    is approximated with a square root function.
    Note: sometimes algorithms return the "log-evidence", or ln(z). The Bayes factor is z.
    Source: Benneke et al. 2013 https://iopscience.iop.org/article/10.1088/0004-637X/778/2/153
    Molliere part by Paul Mollière.
    :param bayes_factor: Bayes factor (aka "evidence")
    :return: sigma significance
    """
    molliere_threshold = 72004899337  # ~exp(25), roughly where numerical errors start to be significant
    is_scalar = False

    if not hasattr(bayes_factor, '__iter__'):
        is_scalar = True
        bayes_factor = np.array([bayes_factor])
    elif not isinstance(bayes_factor, np.ndarray):
        bayes_factor = np.array(bayes_factor)

    benneke_part = np.nonzero(np.less_equal(bayes_factor, molliere_threshold))
    molliere_part = np.nonzero(np.greater(bayes_factor, molliere_threshold))

    sigma = np.zeros(np.shape(bayes_factor))

    if np.size(benneke_part) > 0:
        a = -1 / (np.exp(1) * bayes_factor[benneke_part])
        rho = np.real(np.exp(lambertw(a, -1)))  # the solution to x * log(x) = a is the lambert W function of a
        sigma[benneke_part] = np.sqrt(2) * erfinv(1 - rho)

    if np.size(molliere_part) > 0:
        a = -1 / (np.exp(1) * molliere_threshold)
        rho = np.real(np.exp(lambertw(a, -1)))  # the solution to x * log(x) = a is the lambert W function of a
        offset = np.sqrt(2) * erfinv(1 - rho) - np.sqrt(2 * np.log(molliere_threshold))

        sigma[molliere_part] = np.sqrt(2 * np.log(bayes_factor[molliere_part])) + offset

    if is_scalar:
        return sigma[0]
    else:
        return sigma


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


def calc_met(f):
    # TODO unused?
    return np.log10((f / (np.sum(1e1 ** _get_log_gs()) + f * np.sum(1e1 ** _get_log_mets())))
                    / (1. / (np.sum(1e1 ** _get_log_gs()) + np.sum(1e1 ** _get_log_mets()))))


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


def feature_scaling(array, min_value=0, max_value=1):
    """Bring all values of array between a min and max value.

    Args:
        array: array to normalize
        min_value: target minimum value
        max_value: target maximum value

    Returns:
        The normalized array with values between min_value and max_value.
    """
    return min_value + ((array - np.min(array)) * (max_value - min_value)) / (np.max(array) - np.min(array))


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


def normalize(array, axis=None):
    return (array - np.min(array, axis=axis)) / (np.max(array, axis=axis) - np.min(array, axis=axis))


def plot_radtrans_opacities(radtrans, species, temperature, pressure_bar, mass_fractions=None, co_ratio=0.55,
                            log10_metallicity=0., return_opacities=False, **kwargs):
    import matplotlib.pyplot as plt
    import petitRADTRANS.physical_constants as cst

    def __compute_opacities(_pressures, _temperatures):
        """ Method to calculate and return the line opacities (assuming an abundance
        of 100% for the individual species) of the Radtrans object. This method
        updates the line_struc_kappas attribute within the Radtrans class. For the
        low resolution (`c-k`) mode, the wavelength-mean within every frequency bin
        is returned.

            Args:
                _temperatures:
                    the atmospheric temperature in K, at each atmospheric layer
                    (1-d numpy array, same length as pressure array).

            Returns:
                * wavelength in cm (1-d numpy array)
                * dictionary of opacities, keys are the names of the line_species
                  dictionary, entries are 2-d numpy arrays, with the shape
                  being (number of frequencies, number of atmospheric layers).
                  Units are cm^2/g, assuming an absorber abundance of 100 % for all
                  respective species.

        """

        # Function to calc flux, called from outside
        _opacities = radtrans._interpolate_species_opacities(
            pressures=_pressures,
            temperatures=_temperatures,
            n_g=radtrans.lines_loaded_opacities['g_gauss'].size,
            n_frequencies=radtrans.frequencies.size,
            line_opacities_grid=radtrans.lines_loaded_opacities['opacity_grid'],
            line_opacities_temperature_profile_grid=radtrans.lines_loaded_opacities['temperature_profile_grid'],
            has_custom_line_opacities_tp_grid=radtrans.lines_loaded_opacities['has_custom_tp_grid'],
            line_opacities_temperature_grid_size=radtrans.lines_loaded_opacities['temperature_grid_size'],
            line_opacities_pressure_grid_size=radtrans.lines_loaded_opacities['pressure_grid_size']
        )

        _opacities_dict = {}

        weights_gauss = radtrans.lines_loaded_opacities['weights_gauss'].reshape(
            (len(radtrans.lines_loaded_opacities['weights_gauss']), 1, 1)
        )

        for i, s in enumerate(radtrans.line_species):
            _opacities_dict[s] = np.sum(_opacities[:, :, i, :] * weights_gauss, axis=0)

        return cst.c / radtrans.frequencies, _opacities_dict

    temperatures = np.array(temperature)
    pressure_bar = np.array(pressure_bar)

    temperatures = temperatures.reshape(1)
    pressure_bar = pressure_bar.reshape(1)

    pressures = pressure_bar * 1e6

    wavelengths, opacities = __compute_opacities(pressures, temperatures)
    wavelengths *= 1e4  # cm to um

    opacities_weights = {}

    if mass_fractions is None:
        for s in species:
            opacities_weights[s] = 1.
    elif mass_fractions == 'eq':
        from .poor_mans_nonequ_chem import interpol_abundances

        mass_fractions = interpol_abundances(
            COs_goal_in=co_ratio * np.ones_like(temperatures),
            FEHs_goal_in=log10_metallicity * np.ones_like(temperatures),
            temps_goal_in=temperatures,
            pressures_goal_in=pressure_bar
        )

        for s in species:
            opacities_weights[s] = mass_fractions[s.split('_')[0]]
    else:
        for s in species:
            opacities_weights[s] = mass_fractions[s]

    if return_opacities:
        opacities_dict = {}

        for s in species:
            opacities_dict[s] = [
                wavelengths,
                opacities_weights[s] * opacities[s]
            ]

        return opacities_dict
    else:
        for s in species:
            plt.plot(
                wavelengths,
                opacities_weights[s] * opacities[s],
                label=s,
                **kwargs
            )


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
    rebinned_spectrum = frebin.rebin_spectrum(input_wavelengths, input_spectrum, rebinned_wavelengths)

    if np.all(rebinned_spectrum == -1):
        raise ValueError("something went wrong during re-binning (rebin.f90), check the previous messages")
    elif np.any(rebinned_spectrum < 0):
        raise ValueError(f"negative value in re-binned spectrum, this may be related to the inputs "
                         f"(min input spectrum value: {np.min(input_spectrum)}, "
                         f"min re-binned spectrum value: {np.min(rebinned_spectrum)})")

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


def sigma2bayes_factor(sigma):
    """
    Convert a sigma significance into a Bayes factor, or "evidence".
    Note: sometimes algorithms return the "log-evidence", or ln(z). The Bayes factor is z.
    Source: Benneke et al. 2013 https://iopscience.iop.org/article/10.1088/0004-637X/778/2/153
    :param sigma: sigma significance
    :return: Bayes factor (aka "evidence")
    """
    rho = 1 - erf(sigma / np.sqrt(2))
    return - 1 / (np.exp(1) * rho * np.log(rho))
