============
Contributing
============
We welcome contributions to petitRADTRANS! If you want to contribute a new capability to the code, especially if it is something more involved, please notify the development team before. This is such that we can gauge whether your contribution can or should be merged into pRT.

Suggestions and reporting issues
================================
We always aim to enhance petitRADTRANS, so do not hesitate to report bugs or propose new features or other suggestions. The preferred way in both cases is to create a new issue on the `petitRADTRANS gitlab <https://gitlab.com/mauricemolli/petitRADTRANS/-/issues>`_. In that case, please take the time to read the guidelines below.

While reporting the issue on Gitlab is the preferred method, as it makes the issue public and thus can help other users to solve it, you can directly contact members of the development team `via e-mail <../index.html#contact>`_, if you would like some help with modelling, for example.

Guidelines to report an issue
-----------------------------
- Ensure that the issue is petitRADTRANS-related (see below).
- Use a clear title for the issue.
- Add a few sentences to describe your issue in more details if necessary.
- Add relevant information (OS, pRT version, Python version, use of ``venv`` or ``conda``, etc.).
- Add some key-points: steps to reproduce, expected results and actual results.
- Always include an example that can be used by the developers to reproduce your issue.
- For large scripts or console outputs (more than ~20 lines), please use attached files.
- You may also add how severe you estimate the issue is.

You can check `this example <https://gitlab.com/mauricemolli/petitRADTRANS/-/issues/88>`_ to help you structure your issue.

.. important:: If your issue is resolved (by yourself or a developer), **please report it**, especially if you solved the issue yourself: this can help other people and also the developer team.

While we are happy to help everyone in need, please only report direct petitRADTRANS-related issues. Most of the time issues from, for example, setups (Mac, Conda, compilers, ...) are not related to and cannot be fixed by petitRADTRANS. For these issues we recommend Q&A platforms such as `stackoverflow <https://stackoverflow.co/>`_. If you still encounter difficulties, contact the petitRADTRANS team and we will do our best to help you.

Adding new opacities
====================
You can add new opacities following the instruction in the :doc:`corresponding section <adding_opacities>`.

If you add opacities which are not available through the ExoMol website, and if you are particularly nice, you can share these opacities with us. We would then make them available to the other petitRADTRANS users via this website, while properly attributing your contribution.

Contribute code
===============
Development setup and how to submit changes
-------------------------------------------

.. important:: If you plan for a large addition that takes a lot of time to develop, please inform the development team early. This is to ensure that we are aware of your project and can gauge whether it should be incorporated into petitRADTRANS. Please note that we cannot guarantee that external developments will be merged into the package before this vetting process. If we believe that pRT will benefit from your addition, we may even suggest to integrate you into the development team for a better coordination of development efforts.

If you would like to make a fix or add a feature to petitRADTRANS, you may proceed as follows:

1. Install `Git <https://git-scm.com/>`_.
2. Sign up to `Gitlab <https://gitlab.com/>`_.
3. `Fork <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html>`_ petitRADTRANS from the `main repository <https://gitlab.com/mauricemolli/petitRADTRANS>`_.
    .. note:: You may directly create a new branch (step 10) instead of a fork if you are in the pRT dev team.
4. `Clone <https://docs.gitlab.com/ee/user/project/repository/#clone-a-repository>`_ your fork to work locally.
5. Go inside the cloned directory. Add an upstream remote and fetch it with:
    .. code-block:: bash

        git remote add upstream https://gitlab.com/mauricemolli/petitRADTRANS.git
        git fetch upstream

6. Set your ``main`` branch to track upstream using:
    .. code-block:: bash

        git branch -u upstream/master master


    .. note:: Points 5 and 6 ensure that your fork stays connected with the "official" pRT repository and that you can always incorporate pRT changes into your fork by typing e.g. ``git merge upstream/master``.

7. Follow the petitRADTRANS installation instructions to install your fork. Once you are setup, use the following command:
    .. code-block:: bash

        pip install -e .[test] --no-build-isolation
8. Install `tox <https://tox.wiki/>`_.
9. Check that the test suite is working by executing ``tox`` in the main directory of you fork.
10. Create a new branch using ``git switch --create branch-name``.
11. Make your changes.
12. Regularly commit your changes using ``git commit -m 'Concise description of the change'``.
13. Before pushing, **always** test your changes by executing ``tox``.
14. Push to your branch using ``git push``.

The final step of this process is to create a `merge request <https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html>`_ from your fork, targeting the upstream. Before proceeding, be sure to do the following:

1. Check that you respect the :ref:`stylistic guidelines<stylistic_guidelines>`.
2. Check the :ref:`merge request guidelines<merge_guidelines>`.

You may contact by mail members of the development team to inform them about your (future) merge request at any time in the development process. Stay available in case modifications are requested by the development team before merging your branch.

.. tip:: Make atomic Git commits, accompanying comments should be short but descriptive, starting with a verb in the infinitive.

.. _main_workflow:

pRT dev team: main repository workflow
--------------------------------------

.. image:: images/development_flow.drawio.svg
   :width: 600

The above figure represents the workflow when working on pRT's main repository.

The main repository is composed of at least two branches:

- The ``master`` branch, which contains the stable, latest official release of pRT. It is intended for the users.
- The ``dev`` branch, which contains the latest updates of the package. It is intended for pRT's developers.

Working with the dev branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Any new feature or minor bug fix must always be developed starting from the ``dev`` branch. It is possible to work directly on the ``dev`` branch (locally), or to create a new branch from ``dev``.

When you estimate that your work is done, you can think about pushing to the ``dev`` branch. Before doing so, be sure to do the following:

1. Check that you respect the :ref:`stylistic guidelines<stylistic_guidelines>`.
2. Check the :ref:`merge request guidelines<merge_guidelines>`.
3. Update the ``dev`` version number (see :ref:`versioning`) in the following files:
    - ``pyproject.toml``
    - ``meson.build``
    - ``CHANGELOG.md``
    - ``docs/conf.py``
4. In ``CHANGELOG.md``, update the date of the latest version.
5. Summarize your changes in ``CHANGELOG.md``, respecting the convention (see :ref:`versioning`).
6. You are ready to push!

.. tip:: To ensure the smoothest possible workflow, regularly push your local changes to the remote ``dev`` branch, and regularly update/merge your local version with the remote ``dev``. This way, even major changes can be quickly taken into account, with as few conflicts as possible.

Major bug fixes
~~~~~~~~~~~~~~~
When a major bug (crash of a major function, incorrect results) is identified, the master branch can be directly modified. The patch version of the code must be updated, as well as all development branches. A temporary ``hotfix`` branch can be created if the bug is particularly severe and requires a lot of work.

.. tip:: Make use of :ref:`automatic tests<test_suite>` to prevent such a case to happen!

Backward-compatibility breaking changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When a change breaks backward-compatibility (i.e. changing the name of an attribute, removing a function's argument), the ``dev`` branch itself can be used. This ensures that any new developed feature takes the latest backward-incompatible changes into account. This however slows down or make practically impossible the implementation of new backward-compatible features until the new major version is released.

.. tip:: Do not refrain from making small backward-incompatible changes: in most case you can circumvent the need for a new major version. For example, if you want to change the name of an argument, you can keep the old argument name while adding the new one. Using the old name must still work as expected, but you can make use of ``FutureWarning`` to signal to the users that it is deprecated. You can then push a backward-compatible update, and the old name will be removed in the next major version!

.. _merge_guidelines:

Guidelines before creating a merge request
------------------------------------------
- All code should have :ref:`tests<test_suite>`.
- All code should be documented, functions intended to be used by the user must at least have complete docstrings.
- The test suite (including eventually the tests of your new feature) must raise no error. This include flake8/style errors. You may ask the developer team in case you need help solving these errors.
- The test suite must raise no petitRADTRANS-related warnings. Sometimes warnings may be raised by external libraries, these can be ignored.
- Ensure that you respected the :ref:`stylistic guidelines<stylistic_guidelines>`.

.. _stylistic_guidelines:

Style Guide
-----------
These guidelines are intended to create a consistency within the code, facilitating usage, readability, and maintenance in the long term.

In general, look at existing code for guidance. Before committing, take the time to re-read your code and ensure that you respected the guidelines. Fixing existing code to make it more inline with those guidelines is strongly encouraged.

- It is strongly encouraged to use an IDE such as `PyCharm <https://www.jetbrains.com/pycharm/>`_ (you can use the free community edition) to help you respect the code style.
- Set up your editor to follow `PEP 8 <https://peps.python.org/pep-0008/>`_. In PyCharm, this is enabled by default.
- When implementing an equation or using a specific value, always indicate the source (DOI) in docstrings or comments.
- Respect the `DRY convention <https://en.wikipedia.org/wiki/Don%27t_repeat_yourself#:~:text=%22Don't%20repeat%20yourself%22,redundancy%20in%20the%20first%20place.>`_: **never** copy paste lines of code, create a new function instead if necessary.
- Do not use magic numbers:
    .. code-block:: python

        # Yes
        my_magic_number = 27.849846  # source if relevant, and explanation if a descriptive name is not enough

        if value > my_magic_number:
            ...

        # No
        if value > 27.849846:
            ...
- Name conventions:
    - Use extensive naming: always prefer e.g. ``temperature`` to ``t``. Names should be as descriptive as possible and should be understood **without context**, while reasonably long.
    - Function names in most cases should start with a verb in the infinitive describing the function's purpose.
    - Names for arrays (or lists, dicts, etc.) containing multiple elements must be plural.
    - Use ``get_`` and ``set_`` only for simple functions that perform **no** calculation.
    - Use ``compute_`` for class functions that are static or module functions that calculate something.
    - Use ``calculate_`` only for class functions (first argument ``self``) that calculate something.
    - Use ``<start>2<end>`` for conversions (e.g., from one unit to another as in ``light_year2parsec``).
    - Use ``save`` and ``load`` for I/O functions involving locally stored files.
    - You may begin a function's name with ``from_`` only for ``classmethod``.
    - Functions that are not intended to be used by users, or outside their module/class must start with a single ``_``.
    - Functions that have no purposes outside their context or used only once in the code must start with ``__``.
- Functions in a module or attribute in a class should be declared in alphanumerical order. The character ``_`` is the first character in that order.
- Indented blocks (e.g. ``if``/``else``, ``for``, etc.) should be separated from other code with a blank line (above and below).
    .. code-block:: python

        # Yes
        some_code

        # Eventually, a comment describing what the block is doing
        if condition:
            ...
        else:
            ...

        some_code

        # No
        some_code
        if condition:
            ...
        else:
            ...
        some_code
- Function calls or object instantiations should explicitly display the arguments, one by line, unless there is 1 or less argument or the function is from an external library:
    .. code-block:: python

        # Yes
        function(
            argument_1=value_1,
            argument_2=value_2,
            ...
        )

        # No
        function(value_1, argument_2=value_2,
                 argument_3=value_3, argument_4=value_4,
                 ...)
        function(argument_1=value_1,
                 argument_2=value_2,
                 ...)

        # No (unless the function has 1 or less argument or is from an external library)
        function(argument_1=value_1, argument_2=value_2, ...)
        function(value_1, argument_2=value_2, ...)
        function(value_1, value_2, ...)
- Use parenthesis instead of ``\`` for line breaks:
    .. code-block:: python

        # Yes
        a_very_long_equation = (
            term_1
            * term_2
            * term_3
        )

        # No
        a_very_long_equation = \
            term_1 \
            * term_2 \
            * term_3
- Functions intended to be used by users should have `type hints <https://peps.python.org/pep-0484/>`_.
- Avoid extremely long functions. As a rule of thumb, if a function is more than 100 lines long, break it into smaller functions.
- Docstrings:
    - Must follow the `Google style <https://google.github.io/styleguide/pyguide.html#383-functions-and-methods>`_.
    - Must follow the normal sentence rules.
- Comments (starting with ``#`` on Python):
    - Must never end with a dot.
    - On a line without code, must start with an uppercase.
    - On a line with code, must start with a lowercase.
    - May be exceed the line character limit (120) if they are on a line with code or unbreakable (e.g. URL), in that case add ``# noqa E501`` at the end of the comment to signal ``flake8`` that this is expected.
- If you are using PyCharm, fix all errors, warnings, and weak warnings, with the following exceptions:
    - Errors related to the import of Fortran extensions can be ignored as long as the code works and does not produce warnings.
    - Warnings related to expected types or not found references can be ignored if the warning is related to an external library or a fortran function, as long as the code works and does not produce warnings.
    - Weak warnings related to not using ``kwargs`` can be ignored: this is part of the code architecture.

.. _versioning:

Versioning
----------
petitRADTRANS adheres to `Semantic Versioning <http://semver.org>`_.

Alpha (``X.Y.ZaW``), beta (``X.Y.ZbW``), and release candidate (``X.Y.ZrcW``) versions are used exclusively in the ``dev`` branch.

- Alpha versions: new features are planned or expected to be implemented.
- Beta versions: only bug fixes are planned. No new feature can be added.
- Release candidate versions: only tests are planned. The goal of these versions is to ensure that no hidden bug is left. As for the beta versions, no new feature can be added.

The code's version must be updated in the following files:

- CHANGELOG.md
- meson.build
- pyproject.toml
- docs/conf.py

petitRADTRANS comes with a changelog that is regularly updated with the most notable changes from the code. The format is based on `Keep a Changelog <http://keepachangelog.com>`_.

For items in the "Added" (and "Removed") section, build your sentence as if it started with "Added:" (or "Removed":):
    - Good example: "Function ``my_cool_function`` to do this useful thing.".
    - Bad example: "It is now possible to do this useful thing.".

For items in the "Changed" section, build your sentence as if you answered the question "what has changed?".

For items in the "Fixed" section, build your sentence as if they were the title of an issue (answer to the question: "what issue is fixed?"), and describe the bug that is fixed:
    - Good example: "<A bad thing> happens when function ``definitely_not_my_function`` is called.".
    - Bad example: "Fixes a bug with <some feature>.".

In the changelog, changes are ordered by perceived importance for the user. Changes or fixes internal to an alpha or beta version are not indicated.

.. tip:: Note to devs: the CHANGELOG is primarily destined to the users. They use it to stay informed about new updates and fixes. Conciseness and precision are key to a good changelog!

.. _test_suite:

The petitRADTRANS test suite
----------------------------
How to run the tests?
~~~~~~~~~~~~~~~~~~~~~
1. Install `tox <https://tox.wiki/>`_.
2. Within the petitRADTRANS root directory, execute the command ``tox`` in a console.

All the tests within the "tests" directory will be executed. A summary will be available at the end of the procedure, including test code coverage.

.. important:: Before a push, ``tox`` should always be run.

Conda: running the tests
~~~~~~~~~~~~~~~~~~~~~~~~
When using conda environments, you must install ``tox-conda`` prior to running the tox test suite.

Additionally, you may run into issues with package versions and getting the test suite to run properly, in particular with the ``numba`` package.
You will need to ``conda install numba``, even if you have already installed the package through ``pip``.

We also suggest running ``tox`` for specific python versions, rather than automatically running on the base version installed on your system.
At the very least, you should run tests on the oldest version currently supported by pRT (python 3.9 as of 2024), as well as the most recent version.

Below an example to ``tox`` test the code with ``flake8`` and python 3.11:

.. code-block::

    conda create --name toxfun python=3.11
    conda activate toxfun
    pip install tox
    pip install tox-conda
    conda install numba
    tox -e flake8
    tox -e py311

Introduction
~~~~~~~~~~~~
The tox `configuration <https://tox.wiki/en/latest/config.html>`_ is set within the "setup.cfg" file.

Tox will automatically execute any function in any module across all the project whose name is starting with ``test_``. To keep the code clean, the tests should by default be put within the "tests" directory. The structure of this directory is as follows:
    .. code-block::

        .
        ├── data                           <- contains the test parameters
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
        ├── <numerous test modules>        <- modules containing the testing functions
        └── utils.py                       <- module containing initialisation and comparison functions

The test functions that will be executed by tox are in the test modules. To minimize loading time, there is one module per required ``Radtrans`` object. Other test modules are here to sort tests. Most modules import module ``context`` to ensure that the local version of the code is tested. There is also a ``relative_tolerance`` variable set at the beginning to indicate the relative tolerance when comparing the results with the last validated ones.

In order to keep things clean in the long run: if a test goes wrong, **avoid increasing the tolerance**. Instead, try first to understand the origin of the difference. It is your responsibility as a developer to understand and explain changes in results coming from the changes you made within the code.

Most of the tests consist of calling a petitRADTRANS function, and to compare the result with the last validated one. If an ``AssertionError`` is raised, an error file is automatically generated in the "errors" directory. The error file is a .h5 file containing 6 datasets:

- ``test_outputs``, a ``list`` containing the results of the current test,
- ``reference_outputs``, a ``list`` containing the results of the last validated test,
- ``prt_version``, the version of petitRADTRANS used to generate the last validated test,
- ``relative_tolerance``, the relative tolerance used to compare the results,
- ``absolute_tolerance``, the absolute tolerance used to compare the results.
- ``date``, the date of the test.

In addition, the test will compare the inputs of the tested function. If a discrepancy is detected, an error file is automatically generated in the "errors" directory. The error file is a .h5 file containing 6 datasets:

- ``invalid_test_parameters``, a ``dict`` containing the invalid inputs used for the current test,
- ``reference_parameters``, a ``dict`` containing the corresponding inputs used for the last validated test,
- ``prt_version``, the version of petitRADTRANS used to generate the last validated test,
- ``relative_tolerance``, the relative tolerance used to compare the results,
- ``absolute_tolerance``, the absolute tolerance used to compare the results.
- ``date``, the date of the test.

These files can be used for diagnostic. To load an error file, you can use:

.. code-block:: python

    from tests.benchmark import TestFile

    error_file = TestFile.load('./tests/errors/<filename>.h5')

    # Examples
    # For AssertionError files
    error_file.test_outputs[<output_index>] - error_file.reference_outputs[<output_index>]

    # For invalid inputs files
    error_file.invalid_test_parameters['<parameter_name>'] - error_file.reference_parameters['<parameter_name>']


Creating a new test
~~~~~~~~~~~~~~~~~~~
Tests are used both to ensure that every functionality of the code work, but also that they work **as expected**. It follows that a proper test should:

- Ensure that a function runs.
- Ensure that the results from the function is what is expected.
- Provides an easy way to check the results if they are not expected, and to track the changes that could have led to this discrepancy.
- Be easily reproducible.
- Be as fast as possible without compromising with functionality testing.

In order to create a test, you can use the petitRADTRANS tools and follow these steps:

1. If you need a ``Radtrans`` object (or equivalent), first check if there is one that already suits your need in the existing test modules.
2. If relevant, create a new test module, beginning with ``test``. At the top of the module, put:
    .. code-block:: python

        from .benchmark import Benchmark
        from .context import petitRADTRANS
3. Create your test function (starting with ``test_``). Be as expansive as possible when choosing the name, to make it easier to understand what went wrong if it fails. For the same reason, most of the time you would want to have one functionality tested per test function. The function should have no arguments.
4. Add lines to compare your results with previous ones. To do so, it is highly recommended to use the following structure:
    .. code-block:: python

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
5. Check the dictionary within ``utils.make_petitradtrans_test_config_file`` and look for parameters that you can use in your test function, **if possible without editing them**. If necessary, add key/value pairs to this dictionary. The added values should be small (i.e. no size 10+ array). In general, keep your inputs as small as possible to make tests faster and limit data storage on git. Any larger input (max ~100 kB) should be stored outside this file in the "data" directory. Exception is made for files inside the petitRADTRANS "input_data" directory, that must not be stored on the git.
6. In a python console, execute:
    .. code-block:: python

        from tests.test_my_new_module import test_my_feature  # this will automatically re-generate the parameter file if needed
        Benchmark.activate_reference_file_generation()
        test_my_feature()  # generate the reference comparison file, then test the function
        Benchmark.deactivate_reference_file_generation()
7. Launch ``tox`` to be sure that everything went right.

.. tip:: If your test failed with ``tox``:

    - You can execute your test function in a Python console to help you debug it faster.
    - If you used the recommended ``Benchmark`` workflow, you can also use the generated error files to help you.
    - Error files and reference files have their own class that can be accessed with ``from tests.benchmark import *`` (see points below).
    - You can load ``Benchmark`` error files with ``error_file = TestFile.load('path/to/error_file.h5')``.
    - You can load ``Benchmark`` reference files with ``reference_file = ReferenceFile.load('path/to/reference_file.h5')``.

Resetting all reference files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In rare cases, for example when pushing a new major version, it might be interesting to reset all reference files.
This operation should not be taken lightly as this can have significant consequences on the code's reproducibility and behaviour.
To easily do this operation, execute the following:

.. code-block:: python

    from tests.benchmark import Benchmark
    Benchmark.write_all_reference_files()

Before the reset, you will go through a checklist. Please take the time to read it. If you do not meet all the criteria, cancel the operation.

The petitRADTRANS build Docker image
------------------------------------
In order to speed-up the CI/CD pipeline, the petitRADTRANS' Gitlab repository uses a Docker image stored on Gitlab's `container registry <https://docs.gitlab.com/ee/user/packages/container_registry/>`_. This image is based on a lightweight OS and contains all the libraries necessary to build, test, and deploy petitRADTRANS.

However, to keep pRT up-to-date, it is necessary to regularly update the image. This can be done by running the ``docker/build.sh`` shell script. This script will automatically build a new Docker image using ``docker/Dockerfile``, and push it to Gitlab's container registry, where it will be ready to use by the CI/CD pipeline.
