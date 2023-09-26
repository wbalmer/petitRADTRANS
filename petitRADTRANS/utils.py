"""Stores useful generic functions.
"""
import copy
import warnings

import h5py
import numpy as np


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
        if np.ndim(array) <= 1:
            return array

        array = np.concatenate(array)

    return array


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


def savez_compressed_record(file, numpy_record_array):
    """Apply numpy.savez_compressed on a record array."""
    data_dict = {key: numpy_record_array[key] for key in numpy_record_array.dtype.names}
    np.savez_compressed(file, **data_dict)
