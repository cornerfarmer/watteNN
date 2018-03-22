from setuptools import setup
import Cython
from Cython.Build import cythonize
from distutils.extension import Extension
Cython.Compiler.Options.annotate = True
import glob

extensions = []

def add_extensions(path):
    for file in glob.glob(path + "/*.pyx"):
        extensions.append(Extension(file.replace('/', '.')[:file.find('.pyx')], [file], language="c++", libraries=["profiler"], extra_compile_args=["-std=c++11"], define_macros=[('CYTHON_TRACE', '1')],  include_dirs=[".", "/home/domin/.local/lib/python3.6/site-packages"]))

add_extensions('src')
add_extensions('tests')

setup(
    name="watteNN",
    ext_modules=cythonize(extensions),
    packages=['src', 'tests'],
    package_data={'src': ['*.pxd', '*.pyx']}
)
