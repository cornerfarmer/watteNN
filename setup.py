from setuptools import setup
import Cython
from Cython.Build import cythonize
from distutils.extension import Extension
Cython.Compiler.Options.annotate = True
import glob
import numpy as np
extensions = []

def add_extensions(path):
    for file in glob.glob(path + "/*.pyx"):
        extensions.append(Extension(file.replace('/', '.')[:file.find('.pyx')], [file], language="c++", include_dirs=[np.get_include()]))

add_extensions('src')
add_extensions('tests')

setup(
    name="watteNN",
    ext_modules=cythonize(extensions),
    packages=['src', 'tests'],
    package_data={'src': ['*.pxd', '*.pyx']}
)
