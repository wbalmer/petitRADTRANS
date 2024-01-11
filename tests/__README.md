__# petitRADTRANS test suite Readme
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
├── data                           <- contains the test paramters
|   ├── config_test_radtrans.json  <- the parameters for the test suite
|   ├── <several .npz/.dat files>  <- data files used in tests
|   └── test_stats.json            <- results of the last validated retrieval test
├── errors                         <- if an AssertionError is raised, results will be sent here for diagnostic
├── references                     <- contains the test reference files, storing results from the last validated test
|   └── <numerous .h5 files>       <- results of the last validated retrieval test
├── results                        <- results of the last retrieval test
├── __init__.py                    <- init file (empty)
├── benchmark.py                   <- module containing the Benchmark class, used to compare the results
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
1. If you need a `Radtrans` object (or equivalent), first check if there is one that already suits your need in the existing test modules.
2. If relevant, create a new test module, beginning with "test_". At the top of the module, put: 
    ```
    from .benchmark import Benchmark
    from .context import petitRADTRANS
    ```
3. Create your test function (starting with `test_`). Be as expansive as possible when choosing the name, to make it easier to understand what went wrong if it fails. For the same reason, most of the time you would want to have one functionality tested per test function. The function should have no arguments.
4. Add lines to compare your results with previous ones. To do so, it is highly recommended to use the following structure:
    ```
    def test_my_feature():
        benchmark = Benchmark(
            function=function_to_test,
            relative_tolerance=1e-6
        )
   
        benchmark.run(
            function_to_test_keyword_argument_1=...,
            function_to_test_keyword_argument_2=...,
            ...
        )
    ```
5. Check the dictionary within `utils.make_petitradtrans_test_config_file` and look for parameters that you can use in your test function, **if possible without editing them**. If necessary, add key/value pairs to this dictionary. The added values should be small (i.e. no size 10+ array). In general, keep your inputs as small as possible to make tests faster and limit data storage on git. Any larger input (max ~100 kB) should be stored outside this file in the "data" directory. Exception is made for files inside the petitRADTRANS "input_data" directory, that should not be stored on the git.
6. In a python console, execute:
    ```
    from tests.test_my_new_module import test_my_feature  # this will automatically re-generate the parameter file if needed
    Benchmark.activate_reference_file_generation()
    test_my_feature()  # generate the reference comparison file, then test the function
    Benchmark.deactivate_reference_file_generation()
    ```
7. Launch `tox` to be sure that everything went right!

### Resetting all reference files
In rare cases, for example when pushing a new version, it might be interesting to reset all reference files. 
This operation should not be taken lightly as this can have significant consequences on the code's reproducibility and behaviour.
To easily do this operation, execute the following:
    ```
    from tests.benchmark import Benchmark
    Benchmark.write_all_reference_files()
    ```
