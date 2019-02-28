from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os


anaconda_path = "~/anaconda3"
help_pagename = "Finding your Anaconda Python interpreter path"
help_url = "https://docs.anaconda.com/anaconda/user-guide/tasks/integration/python-path/"
if not os.path.exists(os.path.expanduser(anaconda_path)):
    raise FileNotFoundError(
        "Could not find Anaconda installation in the default location. "
        "Manually set the path within the setup script. "
        "(Ref. '{pagename}': {url}).".format(pagename=help_pagename, url=help_url)
    )

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
