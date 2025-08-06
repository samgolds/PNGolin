![logo](PNGolin_banner.png)
# PNGolin
PNGolin is a Python package that computes binned estimators for the real-space 3D power spectrum, bispectrum, and trispectrum from cosmological density fields. PNGolin was initially developed to study the imprints of primordial non-Gaussianity (PNG) on the large-scale structure (LSS) bispectrum and trispectrum. The main reason to use this code is for the trispectrum estimators. See the test notebook (add link once finished). If you use this code in a publication, please consider citing https://arxiv.org/abs/2407.08731.

## Installation
After cloning the repository, PNGolin can be installed by running \texttt{pip install .}. Note that you will probably need to modify the setup.py file to specify the correct include and library directories for Cython for your environment.

## Dependencies
* cython, pyfftw

Additionally, it is useful to have \href{Pylians}{https://pylians3.readthedocs.io/en/master/} installed. This isn't required for the code, but it is useful for reading in density fields.


