from setuptools import setup
import Cython
from Cython.Build import cythonize
from distutils.extension import Extension
Cython.Compiler.Options.annotate = True



setup(
    name="watteNN",
    ext_modules=cythonize([
        Extension("src.LookUp", ["src/LookUp.pyx"], language="c++", extra_compile_args=["-std=c++11"],  include_dirs=[".", "/home/domin/.local/lib/python3.6/site-packages"]),
        Extension("src.MCTS", ["src/MCTS.pyx"], language="c++", extra_compile_args=["-std=c++11"],  include_dirs=[".", "/home/domin/.local/lib/python3.6/site-packages"]),
        Extension("tests.LookUpTest", ["tests/LookUpTest.pyx"], language="c++", extra_compile_args=["-std=c++11"],  include_dirs=[".", "/home/domin/.local/lib/python3.6/site-packages"]),
        Extension("tests.EnvTest", ["tests/EnvTest.pyx"], language="c++", extra_compile_args=["-std=c++11"],  include_dirs=[".", "/home/domin/.local/lib/python3.6/site-packages"])
    ]),
    packages=['src', 'tests'],
    package_data={'src': ['*.pxd', '*.pyx']}
)
