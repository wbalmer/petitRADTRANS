"""Stores Benchmark and related objects.
Used to test functions and compare the results with saved reference results.
Does not test for performances.
"""
import datetime
import importlib.util
import inspect
import os
import sys

import h5py

from .context import petitRADTRANS
from .utils import tests_references_directory, tests_error_directory


class TestFile:
    def __init__(self, absolute_tolerance: float = None, relative_tolerance: float = None):
        """Base class for test files.
        Can be loaded and saved. Contains the test tolerances, the petitRADTRANS version and the date of the test.

        Args:
            absolute_tolerance: absolute tolerance of the test
            relative_tolerance: relative tolerance of the test
        """
        self.prt_version = petitRADTRANS.__version__
        self.date = None
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance

    def __check_completeness(self, loaded_attributes: dict) -> None:
        """Check if every attribute of the TestFile has been loaded.
        An error is raised if an attribute is missing in the loaded attribute.

        Args:
            loaded_attributes: loaded attributes
        """
        attributes = self.__get_attributes()
        attributes = list(attributes.keys())

        missing_attributes = [
            loaded_attribute for loaded_attribute in loaded_attributes
            if loaded_attribute not in attributes
        ]

        if len(missing_attributes) > 0:
            n_missing_attributes = len(missing_attributes)
            missing_attributes = "'" + ", ".join(missing_attributes) + "'"

            raise KeyError(f"loaded file is missing {n_missing_attributes} attributes: {missing_attributes}")

    def __get_attributes(self) -> dict:
        """Get the public non-callable attributes of the TestFile.

        Returns:
            A dictionary containing all public non-callable attributes of the TestFile
            The keys are the attributes name and the values the attributes value
        """
        attributes = inspect.getmembers(self, lambda a: not inspect.isroutine(a))
        attributes = {
            a[0]: self.__getattribute__(a[0])
            for a in attributes if not a[0].startswith('_')
        }

        return attributes

    def __set_date(self, force_update: bool = False) -> None:
        """Set the date attribute to "now".

        Args:
            force_update: if True, force the date to be updated even if it has already been set
        """
        if self.date is None or force_update:
            self.date = datetime.datetime.now(datetime.UTC).isoformat()

    @classmethod
    def load(cls, file: str):
        """Load a TestFile from an HDF5 file.
        An error is raised if the loaded file misses some attributes or have extra attributes.

        Args:
            file: file to load

        Returns:
            A TestFile.
        """
        with h5py.File(file, "r") as f:
            loaded_attributes = petitRADTRANS.utils.hdf52dict(f)

        new_test_file = cls()

        new_test_file.__check_completeness(loaded_attributes)

        for attribute, value in loaded_attributes.items():
            new_test_file.__setattr__(attribute, value)

        return new_test_file

    def save(self, file: str, rewrite: bool = False, **kwargs) -> None:
        """Save the TestFile in a HDF5 file.

        Args:
            file: file in which to save the TestFile
            rewrite: if True, rewrite the TestFile file if it already exists
            **kwargs: extra parameters to save in the TestFile
        """
        # Set the date to "now"
        self.__set_date()

        # Get all the attributes of the class to initialize the output directory
        output_dict = self.__get_attributes()

        for kwarg, value in kwargs.items():
            output_dict[kwarg] = value

        # Write the save file
        if os.path.isfile(file) and not rewrite:
            raise FileExistsError(f"file '{file}' already exists; set rewrite to True to overwrite this file")

        with h5py.File(file, "w") as f:
            petitRADTRANS.utils.dict2hdf5(
                dictionary=output_dict,
                hdf5_file=f
            )


class ReferenceFile(TestFile):
    parameters_absolute_tolerance = 0.
    parameters_relative_tolerance = 10 ** -sys.float_info.dig

    def __init__(self, parameters: dict = None, outputs: dict = None,
                 absolute_tolerance: float = None, relative_tolerance: float = None):
        super().__init__(absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance)
        self.parameters = parameters
        self.outputs = outputs


class Benchmark:
    _make_reference_file = False
    _reference_file_rewrite = False

    def __init__(self, function: callable, absolute_tolerance: float = 0., relative_tolerance: float = 1e-6,
                 name: str = None):
        if name is None:
            name = inspect.currentframe().f_back.f_code.co_name  # the name of the function that instantiated Benchmark

        self._name = name
        self._function = function
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance

    @property
    def absolute_tolerance(self):
        return self._absolute_tolerance

    @property
    def function(self):
        return self._function

    @property
    def name(self):
        return self._name

    @property
    def relative_tolerance(self):
        return self._relative_tolerance

    def _check_parameters(self, reference_file: ReferenceFile, test_parameters: dict) -> None:
        """Check if the ReferenceFile parameters are the same as the test parameters.

        Args:
            test_parameters:

        Returns:

        """
        invalid_parameters = {}

        for parameter, value in reference_file.parameters.items():
            if parameter not in test_parameters:
                raise KeyError(f"reference file parameter '{parameter}' not in external parameters dict")

            try:
                petitRADTRANS.utils.check_all_close(
                    value,
                    test_parameters[parameter],
                    atol=reference_file.parameters_absolute_tolerance,
                    rtol=reference_file.parameters_relative_tolerance
                )
            except AssertionError:
                invalid_parameters[parameter] = value

        if len(invalid_parameters) > 0:
            self._write_error_file(
                file_suffix='invalid_parameters',
                error_dict={
                    'invalid_test_parameters': {
                        parameter: value
                        for parameter, value in test_parameters.items() if parameter in invalid_parameters
                    },
                    'reference_parameters': invalid_parameters
                },
                absolute_tolerance=reference_file.parameters_absolute_tolerance,
                relative_tolerance=reference_file.parameters_relative_tolerance
            )

            invalid_parameters = "'" + '", "'.join(invalid_parameters) + "'"

            raise AssertionError(f"some parameters in reference file were not identical to the test parameters: "
                                 f"{invalid_parameters}\n"
                                 f"The invalid parameters has been saved for diagnostic")

    def _get_function_arguments(self, parameters: dict) -> dict:
        # Get the function arguments
        arguments_names = self.function.__code__.co_varnames

        # Check for missing arguments
        missing_arguments = [
            argument_name for argument_name in arguments_names
            if argument_name not in parameters
        ]

        if len(missing_arguments) > 0:
            n_missing_arguments = len(missing_arguments)
            missing_arguments = "'" + ", ".join(missing_arguments) + "'"

            raise ValueError(f"test function is missing {n_missing_arguments} arguments: {missing_arguments}")

        return {
            parameter: value for parameter, value in parameters
            if parameter in arguments_names
        }

    def _get_reference_file(self) -> str:
        return os.path.join(
            tests_references_directory,
            str(self._name) + '_ref.h5'
        )

    def _run(self, **kwargs) -> dict:
        outputs = self.function(**kwargs)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        return {f"{i}": output for i, output in enumerate(outputs)}

    def _write_error_file(self, file_suffix: str, error_dict: dict,
                          absolute_tolerance: float, relative_tolerance: float) -> None:
        error_file = TestFile(
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

        error_filename = os.path.join(
            tests_error_directory,
            str(self._name) + '_' + file_suffix + '.h5'
        )

        print(f"Saving error file '{error_filename}'...")

        if not os.path.isdir(tests_error_directory):
            os.mkdir(tests_error_directory)

        error_file.save(
            file=error_filename,
            rewrite=True,
            **error_dict
        )

    @staticmethod
    def activate_reference_file_generation() -> None:
        Benchmark._make_reference_file = True
        print(f"Benchmark reference file generation has been activated")

    @staticmethod
    def make_all_reference_files(test_directory=None):
        if test_directory is None:
            test_directory = os.path.abspath(os.path.dirname(__file__))

        # Activate reference file making
        Benchmark.activate_reference_file_generation()

        try:
            # Get all test files in the "tests" directory
            test_directory = os.path.abspath(test_directory)
            test_files = [
                os.path.join(test_directory, f)
                for f in os.listdir(test_directory)
                if os.path.isfile(os.path.join(test_directory, f))
            ]
            test_files = [
                test_file
                for test_file in test_files
                if os.path.basename(test_file).startswith('test_') and os.path.basename(test_file).endswith('.py')
            ]

            for i, test_file in enumerate(test_files):
                # Get the test file module name
                module_name = os.path.basename(test_file).rsplit('.py', 1)[0]  # remove extension
                module_name = "tests." + module_name

                # Import the test file
                print(f"Importing test module from file '{test_file}' ({i + 1}/{len(test_files)})...")
                spec = importlib.util.spec_from_file_location(module_name, test_file)
                test_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = test_module
                spec.loader.exec_module(test_module)

                test_functions = dir(test_module)

                for test_function_name in test_functions:
                    test_function = test_module.__getattribute__(test_function_name)

                    if test_function_name.startswith('test_') and callable(test_function):
                        print(f" Running '{module_name}.{test_function_name}'...")
                        test_function()
                    else:
                        print(f" Ignored non-test function '{test_function_name}'")

            print(f"Successfully made all test reference files")
        finally:
            Benchmark.reset_reference_file_generation()

    @staticmethod
    def reset_reference_file_generation() -> None:
        Benchmark._make_reference_file = False
        Benchmark._reference_file_rewrite = False
        print(f"Benchmark reference file generation state has been reset to default")

    def run(self, **kwargs) -> None:
        if Benchmark._make_reference_file:
            self.write_reference_file(**kwargs)

        self.test(**kwargs)

    def test(self, **kwargs) -> None:
        reference_file = self._get_reference_file()

        if not os.path.isfile(reference_file):
            raise FileNotFoundError(f"reference file '{reference_file}' does not exist, "
                                    f"generate it first to run the test")

        reference_file = ReferenceFile.load(reference_file)

        print(f"Comparing '{self._name}' results "
              f"from petitRADTRANS-{reference_file.prt_version} ({reference_file.date}) "
              f"and petitRADTRANS-{petitRADTRANS.__version__}...")

        self._check_parameters(
            reference_file=reference_file,
            test_parameters=kwargs
        )

        outputs = self._run(**kwargs)

        try:
            petitRADTRANS.utils.check_all_close(
                outputs,
                reference_file.outputs,
                atol=self._absolute_tolerance,
                rtol=self._relative_tolerance
            )
        except AssertionError:
            self._write_error_file(
                file_suffix='invalid_outputs',
                error_dict={
                    'test_outputs': outputs,
                    'reference_outputs': reference_file.outputs
                },
                absolute_tolerance=self._absolute_tolerance,
                relative_tolerance=self._relative_tolerance
            )

            raise

        print(f"Test successful "
              f"(absolute and relative tolerances: {self._absolute_tolerance}, {self._relative_tolerance})")

    def write_reference_file(self, **kwargs) -> None:
        outputs = self._run(**kwargs)

        reference_file = ReferenceFile(
            parameters=kwargs,
            outputs=outputs,
            absolute_tolerance=self._absolute_tolerance,
            relative_tolerance=self._relative_tolerance
        )

        reference_file_name = self._get_reference_file()

        if os.path.isfile(reference_file_name) and not Benchmark._reference_file_rewrite:
            answer = input(
                f"You are about to rewrite existing test reference files.\n"
                f"You should *not* do this if:\n"
                f" - you have failed comparison tests and have not identified, understood, "
                f"and fixed (if relevant) the reason of these failures,\n"
                f" - you made a change in the code that is not affecting all results and ran make_all_reference_files()"
                f" (in that case, run only the affected tests),\n"
                f" - you want to add a non-existing reference file without changing the current ones.\n"
                f"Do you want to proceed? (Yes/N)"
            )

            if answer != 'Yes':
                print("Aborting test reference file rewrite")
                return

            answer = input(
                f"Changing test reference files can have significant consequences in reproducibility with previous "
                f"versions of the code. "
                f"Hence, before proceeding you need to ensure that some conditions are respected.\n"
                f"*Please type exactly the affirmative answer to proceed*\n"
                f"(1/2) Have you properly documented this change publicly "
                f"in a clear and accessible way (e.g. in the CHANGELOG)? "
                f"(Yes, I have documented this change/N)"
            )

            if answer != 'Yes, I have documented this change':
                print("Please take the time to properly document this change before proceeding")
                return

            answer = input(
                f"(2/2) Have you informed the other developers of this change and do they agree with it? "
                f"(Yes, the other developers agree with the change/N)"
            )

            if answer != 'Yes, the other developers agree with the change':
                print("Please inform the other developers of the change and wait for their approval")
                return

            print(f"Checklist done. Proceeding to test reference file overwrite...")
            Benchmark._reference_file_rewrite = True

        print(f"Saving reference file '{reference_file_name}'...")

        reference_file.save(reference_file_name, rewrite=True)
