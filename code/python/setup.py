from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("mobiquitous_experiment_runner.pyx")
)
