# For debugging: set to debug to 'true' and optimization to 'g'

project('petitRADTRANS',
    'c',
    # version must be updated here as well
    version : '4.0.0a42',
    license: 'MIT',
    meson_version: '>=1.1.0',
    default_options : [
        'debug=false',
        'optimization=3',
        'warning_level=1',
        'c_std=c99'
    ],
)

add_languages('cpp', native: false)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# Locate the XLA FFI headers shipped with jaxlib; the native correlated-k
# mixing kernel is built against them (header-only, no linking against XLA).
# Use relative paths to prevent meson from complaining about absolute paths.
incdir_jax_ffi = run_command(py,
    ['-c',
    'import os; os.chdir(".."); import jax.ffi; print(os.path.relpath(jax.ffi.include_dir(), "./petitRADTRANS"))'],
    check : true
).stdout().strip()

inc_jax_ffi = include_directories(incdir_jax_ffi)

# Native extensions (XLA FFI custom-call kernels). These replace the former
# f2py-built Fortran backend; the Fortran sources are kept in
# petitRADTRANS/fortran_src for reference but are no longer built.
py.extension_module('prt_ck_mixing_ffi',
    'petitRADTRANS/cpp_src/prt_ck_mixing_ffi.cc',
    include_directories: inc_jax_ffi,
    dependencies: py_dep,
    override_options: ['cpp_std=c++17'],
    install: true,
    subdir: 'petitRADTRANS'
)

# Install python sources
subdir('petitRADTRANS')
