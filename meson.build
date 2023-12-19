# For debugging: set to debug to 'true' and optimization to 'g'

project('petitRADTRANS',
    'c',
    # version must be updated here as well
    version : '3.0.0a126',
    license: 'MIT',
    meson_version: '>=1.1.0',
    default_options : [
        'debug=false',
        'optimization=3',
        'warning_level=1'
    ],
)

add_languages('fortran', native: false)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# Include f2py directories for compilation
# Use relative paths to prevent meson from complaining about absolute paths
incdir_numpy = run_command(py,
    ['-c',
    'import os; os.chdir(".."); import numpy; print(os.path.relpath(numpy.get_include(), "./petitRADTRANS"))'],
    check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c',
    'import os; os.chdir(".."); import numpy.f2py; print(os.path.relpath(numpy.f2py.get_include(), "./petitRADTRANS"))'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Set extensions names
extension_names = [
    'fortran_radtrans_core',
    'fortran_inputs',
    'fortran_rebin',
    'fortran_chemistry'
]

# Set extension files
fortran_source_directory = 'petitRADTRANS/fortran_src/'
extensions = {}

foreach extension_name : extension_names
    extensions += {extension_name: fortran_source_directory + extension_name + '.f90'}
endforeach

# Add extensions
foreach extension_name, extension_file : extensions
    extension_module = extension_name + 'module.c'
    extension_source = custom_target(extension_module,
        input : [extension_file],
        output : [extension_module, extension_name + '-f2pywrappers2.f90'],
        command : [py, '-m', 'numpy.f2py', '-m', extension_name, '@INPUT@']
    )  # inform meson that the .f90 file need to be compiled using f2py

    py.extension_module(extension_name,
        [extension_file, extension_source],
        incdir_f2py / 'fortranobject.c',
        include_directories: inc_np,
        dependencies: py_dep,
        install: true,
        subdir: 'petitRADTRANS'
    )
endforeach

# Install python sources
subdir('petitRADTRANS')
