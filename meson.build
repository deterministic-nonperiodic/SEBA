project('seba', 'fortran', 'c',
  version: '0.1.0',
  license: 'MIT',
  meson_version: '>=1.1.0',
  default_options: [
    'warning_level=1',
    'b_fortran_implicit_none=true'
  ]
)

# --- Python and NumPy setup ---
py = import('python').find_installation(pure: false)
numpy_dep = dependency('numpy', required: true)
f2py = find_program('f2py', required: true)

# --- OpenMP support ---
option_openmp = get_option('openmp')
openmp_dep = dependency('openmp', required: option_openmp, language: 'fortran')

# --- Fortran extension: numeric_tools ---
mod_name = 'numeric_tools'
fortran_source = 'src/seba/fortran_libs/numeric_tools.f90'

# infer extension suffix
mod_suffix = run_command(py, '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))').stdout().strip()

numeric_tools = custom_target(
  mod_name,
  input: fortran_source,
  output: mod_name + mod_suffix,
  command: [
    'env',
    'FFLAGS=-fopenmp',
    'LDFLAGS=-fopenmp -lgomp',
    f2py, '-c', '-m', mod_name, '@INPUT@', '--quiet', '--fcompiler=gnu95'
  ],
  install: true,
  install_dir: join_paths(py.get_install_dir(), 'seba')
)


# --- Installing Python sources ---
py.install_sources([
  'src/seba/__init__.py',
  'src/seba/constants.py',
  'src/seba/io_tools.py',
  'src/seba/kinematics.py',
  'src/seba/seba.py',
  'src/seba/spectral_analysis.py',
  'src/seba/spherical_harmonics.py',
  'src/seba/thermodynamics.py',
  'src/seba/tools.py',
  'src/seba/visualization.py',
  'src/seba/inter_scale_transfers.py'
], subdir: 'seba')

# --- Installing data files ---
cm_data_files = files(
  'src/seba/cm_data/bcw.cm',
  'src/seba/cm_data/cet_d02.cm',
  'src/seba/cm_data/cet_d10.cm',
  'src/seba/cm_data/cet_d13.cm'
)

install_data(cm_data_files, install_dir: join_paths(py.get_install_dir(), 'seba', 'cm_data'))

# --- Summary ---
message('✔ Configuration for SEBA complete.')
message('Python version:      ' + py.version())
message('Install location:    ' + py.get_install_dir())
message('NumPy version:       ' + numpy_dep.version())
message('OpenMP enabled:      ' + openmp_dep.found().to_string())

