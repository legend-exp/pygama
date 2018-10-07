#!/usr/bin/env python
from setuptools import setup, Extension, find_packages
import sys,os

do_cython = False
try:
    from Cython.Build import cythonize
    do_cython = True
except ImportError:
    do_cython = False

if __name__ == "__main__":

    try:
        import numpy as np
        include_dirs = [np.get_include(),]
    except ImportError:
        do_cython = False

    src = []
    ext = ".pyx" if do_cython else ".c"
    src += [
        os.path.join("pygama","processing", "_pygama"+ext)
    ]

    ext = [Extension(
            "pygama.processing._pygama",
            sources=src,
            language="c",
            include_dirs=include_dirs
        ),
        Extension(
                "pygama.transforms",
                sources=[os.path.join("pygama", "transforms"+ext)],
                language="c",
            )
        ]

    if do_cython: ext = cythonize(ext)

    setup(
        name="pygama",
        version="0.1.0",
        author="Ben Shanks",
        author_email="benjamin.shanks@gmail.com",
        packages=find_packages(),
        ext_modules=ext,
        install_requires=["numpy", "scipy", "pandas", "tables", "future", "cython"]
    )
