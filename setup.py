from distutils.core import setup
import Cython
from Cython.Build import cythonize
from distutils.extension import Extension
Cython.Compiler.Options.annotate = True



setup(
    name="src",
    ext_modules=cythonize([
        Extension("src.LookUp", ["src/LookUp.pyx"], language="c++", extra_compile_args=["-std=c++11"],  include_dirs=[".", "/home/domin/.local/lib/python3.6/site-packages"]),
        Extension("src.MCTS", ["src/MCTS.pyx"], language="c++", extra_compile_args=["-std=c++11"],  include_dirs=[".", "/home/domin/.local/lib/python3.6/site-packages"])
    ]),
    packages=['src'],
    package_data={'src': ['*.pxd', '*.pyx']}
)
