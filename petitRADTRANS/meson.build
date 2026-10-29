subdir('ccf')
subdir('chemistry')
subdir('cli')
subdir('config')
subdir('opacities')
subdir('plotlib')
subdir('temperature_profiles')
subdir('radtrans_core')
subdir('retrieval')
subdir('slurm_runs')
subdir('stellar_spectra')

py.install_sources([
    'jax_dtype.py',
    'math.py',
    'physical_constants.py',
    'physics.py',
    'planet.py',
    'radtrans.py',
    'spectral_model.py',
    'synthetic_photometry.py',
    'utils.py',
    'validation.py',
    '_input_data.py',
    '__setup_meson_build.py',
    '__init__.py',
    ],
    subdir: 'petitRADTRANS'
)

configure_file(
    input: '_version.py.in',
    output: '_version.py',
    configuration: {'VERSION': meson.project_version()},
    install: true,
    install_dir: py.get_install_dir() / 'petitRADTRANS',
)
