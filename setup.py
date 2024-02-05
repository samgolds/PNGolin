from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from platform import machine, system
import numpy 

################################################################################
# Install by running:
# CC=/opt/homebrew/opt/llvm/bin/clang++ python setup.py build_ext --inplace --force
################################################################################

extra_compile_args=['-fopenmp', '-Ofast'] 
extra_link_args=['-lomp']

ext_modules = [
    Extension(
        'nspectra',
        sources=["nspectra.pyx"],
        extra_compile_args=extra_compile_args,  
        extra_link_args=extra_link_args
    )
]

setup(
    name = "nspectra",
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)