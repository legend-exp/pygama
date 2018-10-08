#!/usr/bin/env python3
""" placeholder for pygama code QC script
    - run pylint over all code: pylint [files]
    - run yapf over all code: yapf --style google -i [files]
    - documentation generated w/ sphinx (worry about web version later, since it may need to be private software?)
    - package installs and uninstalls with pip
    - ignore auto-generated files (makes cython a dependency, but ok)
    - can we add a type checker?
"""
import glob
import subprocess
from pathlib import Path


def main():

    # make a list of python files
    pyfiles = list(Path("./pygama/").rglob("*.py"))
    pyfiles.extend(list(Path("./pygama/").rglob("*.pyx")))
    pyfiles.extend(["./tests.py", "./setup.py"])

    # run the auto-formatter
    for f in pyfiles:
        print("Running yapf on file: {}".format(f))
        sh("yapf --style google -i {}".format(f))


def sh(cmd, sh=False):
    """ Wraps a shell command."""
    import shlex
    import subprocess as sp
    if not sh: sp.call(shlex.split(cmd))  # "safe"
    else: sp.call(cmd, shell=sh)  # "less safe"


if __name__ == "__main__":
    main()
