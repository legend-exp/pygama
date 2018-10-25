#!/usr/bin/env python3
""" pygama code QC suite
    - run pylint over all code: pylint [files]
    - run yapf over all code: yapf --style google -i [files]
    - documentation generated w/ sphinx (worry about web version later, since it may need to be private software?)
    - package installs and uninstalls with pip
    - ignore auto-generated files (makes cython a dependency, but ok)
    - should we add a static type checker?
    - processing speed checks?
"""
import sys, os, glob, argparse
import subprocess
from pathlib import Path


def main(argv):

    opts = argparse.ArgumentParser(description='pygama test suite')
    opts.add_argument('-t','--toggle', help='Toggle cythonizing', action='store_true')
    opts.add_argument('-y','--yapf', help='Run yapf on module', action='store_true')
    opts.add_argument('-l','--lint', help='Run pylint on module', action='store_true')
    args = vars(opts.parse_args())
    if args['lint']:
        pylint()
    elif args['yapf']:
        yapf()
    elif args['toggle']:
        toggle()


def toggle():

    # get source files
    pyxfiles = []
    pyxfiles.extend(glob.glob("./pygama/processing/*.pyx"))
    pyxfiles.extend(glob.glob("./pygama/decoders/*.pyx"))

    pyfiles = []
    pyfiles.extend(glob.glob("./pygama/processing/*.py"))
    pyfiles.extend(glob.glob("./pygama/decoders/*.py"))
    pyfiles = [f for f in pyfiles if "__init__.py" not in f]

    # change to python
    if len(pyxfiles) > 0 and len(pyfiles) == 0:
        for f_name in pyxfiles:
            os.rename(f_name, f_name.replace(".pyx", ".py"))

    # change to cython
    elif len(pyxfiles) == 0 and len(pyfiles) > 0:
        for f_name in pyfiles:
            os.rename(f_name, f_name.replace(".py", ".pyx"))

    else:
        print("Found some dumb mix of files, couldn't toggle.")
        exit()

    # remove any temp cython files
    cfiles = []
    cfiles.extend(list(Path("./pygama/").rglob("*.so")))
    cfiles.extend(list(Path("./pygama/").rglob("*.c")))
    for cf in cfiles:
        os.remove(cf)


def yapf():
    """ run the auto-formatter """
    for f in pyfiles():
        print("Running yapf on file: {}".format(f))
        sh("yapf --style google -i {}".format(f))


def pylint():
    """ run pylint (not tested yet ...) """
    for f in pyfiles():
        print("Running pylint on file: {}".format(f))
        sh("pylint {}".format(f))


def pyfiles():
    """ make a list of python files """
    pyfiles = list(Path("./pygama/").rglob("*.py"))
    pyfiles.extend(list(Path("./pygama/").rglob("*.pyx")))
    pyfiles.extend(["./tests.py", "./setup.py"])
    return pyfiles


def sh(cmd, sh=False):
    """ Wraps a shell command."""
    import shlex
    import subprocess as sp
    if not sh: sp.call(shlex.split(cmd))  # "safe"
    else: sp.call(cmd, shell=sh)  # "less safe"




if __name__ == "__main__":
    main(sys.argv[1:])
