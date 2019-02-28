from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np


anaconda_path = "~/anaconda3"

ext_modules = [
    Extension(
        name="mkl_spblas",
        sources=[
            "mkl_spblas.pyx"
        ],
        include_dirs=[
            anaconda_path + "/include",
            np.get_include(),
        ],
    )
]

setup(
    ext_modules = cythonize(ext_modules, annotate=True)
)
