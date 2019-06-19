#!/usr/bin/env python
""" pygama setup script.
run from containing folder with:
$ pip install -e pygama
re-runs cythonize function on a list of extensions.
"""
from setuptools import setup, Extension, find_packages
import sys, os, glob

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
    fext = ".pyx" if do_cython else ".c"

    exts = []
    for mod in ["decoders","processing"]:
        for f in glob.glob("./pygama/{}/*.pyx".format(mod)):
            f_name = f.split('/')[-1]
            cyname = os.path.splitext(f_name)[0]
            exts.append(Extension(
                "pygama.{}.{}".format(mod, cyname),
                sources = [os.path.join("pygama", mod, cyname + fext)],
                language = 'c',
                include_dirs = include_dirs)
                )

    if do_cython:
        exts = cythonize(exts)

    setup(
        name="pygama",
        version="0.2",
        author="Clint Wiseman",
        author_email="wisecg.neontetra@gmail.com",
        packages=find_packages(),
        ext_modules=exts,
        install_requires=[
            "numpy", "scipy", "pandas", "tables", "future", "cython", "tinydb"
        ])
