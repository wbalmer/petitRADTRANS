# petitRADTRANS test suite Readme
This should probably be merged within a future contributing file.

## How to use the test suite?
1. Install [tox](https://tox.wiki/en/latest/install.html).
2. Within the petitRADTRANS root directory, execute command `tox` in a console.
3. That's it. All the tests within the "tests" directory will be executed. A summary will be available at the end of the procedure.

Ideally, `tox` should be run before committing or pushing.

### A brief summary of the test suite
The tox [configuration](https://tox.wiki/en/latest/config.html) is set within the "setup.cfg" file.

Tox will automatically execute any function in any module across all the project whose name is starting with `test_`. To keep the code clean, the tests should by default be put within the "tests" directory. The structure of this directory is as follows:
```
.
├── data                           <- contains the test setups and the results from the last validated test
|   ├── config_test_radtrans.json  <- the parameters for the test suite
|   ├── <numerous .npz files>      <- results of last validated tests
|   └── test_stats.json            <- results of the last validated retrieval test
├── errors                         <- if an AssertionError is raised, results will be sent here for diagnostic
├── results                        <- results of the last retrieval test
├── __init__.py                    <- init file (empty)
├── context.py                     <- loaded in tests modules in order to ensure that the local version of petitRADTRANS is tested
├── reference_files_generators.py  <- contains functions to generate and save reference files (i.e. the results from the latest validated test)
├── <numerous test modules>        <- modules containing the testing functions
└── utils.py                       <- module containing initialisation and comparison functions
```
The test functions that will be executed by tox are in the test modules. To minimize loading time, there is one module per required `Radtrans` object. Other test modules are here to sort tests. Most modules import module `context` to ensure that the local version of the code is tested. There is also a `relative_tolerance` variable set at the beginning to indicate the relative tolerance when comparing the results with the last validated ones. In order to keep things clean in the long run: if a test goes wrong, **avoid increasing the tolerance**. Instead, try first to understand the origin of the difference. It is your responsibility as a developer to understand and explain changes in results coming from the changes you made within the code.

Most of the tests consist of calling a petitRADTRANS function, and to compare the result with the last validated one. If an AssertionError is raised, an error file is automatically generated in the "errors" directory. The error file is a .npz file containing 4 keys: 
- `test_result`, the result of the current test, 
- `data`, the result of the last validated test, 
- `relative_tolerance`, the relative tolerance used to compare the results, 
- `absolute_tolerance`, the absolute tolerance used to compare the results.

This file can be used for diagnostic.

### Creating a new test
Tests are used both to ensure that every functionality of the code work, but also that they work **as expected**. It follows that a proper test should:
- Ensure that a function runs.
- Ensure that the results from the function is what is expected.
- Provides an easy way to check the results if they are not expected, and to track the changes that could have led to this discrepancy.
- Be easily reproducible.
- Be as fast as possible without compromising with functionality testing.

In order to create a test, you can use the petitRADTRANS tools and follow these steps:
1. If you need a `Radtrans` object, first check if there is one that already suits your need in the existing test modules.
2. If relevant, create a new test module. At the beginning of the module, put: 
    ```
    from .context import petitRADTRANS
    from .utils import compare_from_reference_file
    ```
3. Create your test function (starting with `test_`). Be as expansive as possible when choosing the name, to make it easier to understand what went wrong if it fails. For the same reason, most of the time you would want to have one functionality tested per test function. The function should have no arguments.
4. Add lines to compare your results with previous ones. To do so, it is highly recommended to use the `utils.compare_form_reference_file` function (check the docstrings for more information).
5. Copy your test function in "reference_file_generator.py", but remove the `test_` at the beginning of the name. For clarity and consistency you should replace it with `create_<function_name>_ref`.
6. In the copied function, if relevant import what is necessary from your test module (e.g. a `Radtrans` object used in common). Use a relative import to be sure that you are using your local module.
7. If relevant, you can add a `plot_figure=False` argument to the function.
8. Remove the comparison part of the function (e.g. the call to `compare_from_reference_file`).
9. Add lines to save the results you want to test within a file. The file should be in .npz format (`numpy.savez_compressed()`), be stored within the "data" directory, and have at least keys: 
    - `header`, to indicate how the file was generated and the units of the results if relevant,
    - `prt_version`, to indicate the petitRADTRANS version with which the file has been generated.
    
    These keys serve no other purpose than keeping track of changes and easing debugging. While not mandatory they can save precious time. For that purpose, it is highly recommended to use (or create) a `__save` function in the `reference_files_generators` module.
10. Add a call to your function in function `create_all_comparison_files`, at the end of the `reference_files_generators` module.
11. In the `utils` module, add to the dictionary `reference_filenames` a key/value pair with your function name as key and your file core name (i.e. without path and without extension) + "_ref" as value.
12. Go back to your `create_` and `test_` functions, and replace your filename and reference file with `reference_filenames['your_reference_file_key']`. This is to keep everything consistent and trackable.
13. Check the dictionary within `utils.create_test_radtrans_config_file` and look for parameters that you can use in your test function, **if possible without editing them**. If necessary, add key/value pairs to this dictionary. The added values should be small (i.e. no size 10+ array). In general, keep your inputs as small as possible to make tests faster and limit data storage on git. Any larger input (max ~100 kB) should be stored outside this file in the "data" directory. Exception is made for files inside the petitRADTRANS "input_data" directory, that should not be stored on the git.
14. Go back to your `create_` and `test_` functions, and replace your parameter values with `radtrans_parameters['<parameter_name>']`. This will ensure that the parameters used for the test are without ambiguity the same in the create and test functions. It also stores parameters outside the code, os it prevents a test failure resulting from code edition.
15. Update the `utils.version` value to the current petitRADTRANS version.
16. If you added new parameters, delete the "data/config_test_radtrans.json".
17. In a python console, execute:
    ```
    from tests.reference_files_generators import <new_create_function_name>  # re-generate the parameter file if needed
    <new_create_function_name>()  # generate the reference comparison file
    ```
18. Launch `tox` to be sure that everything went right!