from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import sys

def is_platform_windows():
    return sys.platform == "win32"


def is_platform_mac():
    return sys.platform == "darwin"


def is_platform_linux():
    return sys.platform.startswith("linux")


if is_platform_windows():
    extra_compile_args = ['/openmp']
    extra_link_args = []
else:
    extra_compile_args = ["-fopenmp"]
    extra_link_args = ["-fopenmp"]

# --- pygrid3 compile ---

print(' ')
print('**********************************')
print('******** Building pygrid3 ********')
print('**********************************')
print(' ')

ext_modules = [Extension("main/gen_DENSE/pygrid_internal.c_grid",
                         ["main/gen_DENSE/pygrid_internal/src/c_grid.pyx"],
                         language="c++",
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         include_dirs=["main/gen_DENSE/pygrid_internal/src/", numpy.get_include()],
                         library_dirs=["main/gen_DENSE/pygrid_internal/src/"])]

setup(
  name = 'c_grid',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules 
)