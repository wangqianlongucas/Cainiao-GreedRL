import os
import re
import sys
import time
import subprocess

from distutils import sysconfig
from setuptools import setup, Extension, find_packages
from Cython.Build import build_ext, cythonize


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            if not extdir.endswith(os.path.sep):
                extdir += os.path.sep

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            subprocess.check_call(['cmake', ext.sourcedir, '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir],
                                  cwd=self.build_temp)
            subprocess.check_call(['cmake', '--build', '.', '--', 'VERBOSE=1', '-j8'], cwd=self.build_temp)
        else:
            super().build_extension(ext)


ext_modules = [CMakeExtension('greedrl_c')]

setup(
    name='greedrl',
    version='1.0.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': CMakeBuild},
    install_requires=["torch==1.12.1+cu113"],
)
