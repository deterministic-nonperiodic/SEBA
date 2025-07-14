import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class CustomBuild(build_ext):
    def run(self):
        # Environment flags for OpenMP
        env = os.environ.copy()
        env["FFLAGS"] = "-fopenmp"
        env["LDFLAGS"] = "-fopenmp -lgomp"

        build_cmd = [
            sys.executable,
            "-m", "numpy.f2py",
            "-c", "src/seba/fortran_libs/numeric_tools.f90",
            "-m", "src.seba.numeric_tools",
            "--fcompiler=gnu95",
            "--quiet"
        ]

        subprocess.check_call(build_cmd, env=env)
        super().run()


setup(
    name='seba',
    version='0.1.0',
    license='MIT',
    description='Spectral Energy Budget of the Atmosphere',
    author='Yanmichel A. Morfa',
    author_email="morfa@iap-kborn.de",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    cmdclass={'build_ext': CustomBuild},
    setup_requires=['numpy'],
    install_requires=[
        'numpy',
        'scipy',
        'xarray',
        'matplotlib',
        'pint',
        'shtns'
    ],
    include_package_data=True,
    package_data={
        'seba': ['cm_data/*.cm', 'numeric_tools*.so'],
    },
    zip_safe=False,
)
