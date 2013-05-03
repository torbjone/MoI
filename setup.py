#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("cython_funcs", ["cython_funcs.pyx"], include_dirs=[numpy.get_include()])]

setup(
  name = 'cython_funcs',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
