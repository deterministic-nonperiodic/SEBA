import os
import shutil

from numpy.distutils.core import setup, Extension
from setuptools import find_packages

# Clean build directory to force recompiling fortran libraries...
build_dir = 'build'
if os.path.exists(build_dir):
    shutil.rmtree(build_dir)

setup(
    name='seba',
    version='0.1.0',
    license='MIT',
    description='A package for computing the spectral energy budget of the atmosphere',
    author='Yanmichel A. Morfa',
    author_email='yanmichel.morfa-avalos@mpimet.mpg.de',
    packages=find_packages(),
    package_dir={'seba': 'seba'},
    ext_modules=[Extension('numeric_tools', sources=['seba/fortran_libs/numeric_tools.f90'])],
    python_requires='>=3.9',
    setup_requires=['numpy'],
    install_requires=[
        'pint',
        'shtns',
        'xarray',
        'numpy',
        'scipy',
        'matplotlib'
    ],
    include_package_data=True,
    package_data={'seba': ['*.so']},
    zip_safe=False
)
