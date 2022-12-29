from __future__ import annotations

import fnmatch
import glob
import logging
import os
from typing import Union

import h5py
import numpy as np
import pandas as pd

from .. import Array, Scalar, Struct, VectorOfVectors
from .store import LH5Store

LGDO = Union[Array, Scalar, Struct, VectorOfVectors]

log = logging.getLogger(__name__)


def ls(lh5_file: str | h5py.Group, lh5_group: str = "") -> list[str]:
    """Return a list of LH5 groups in the input file and group, similar
    to ``ls`` or ``h5ls``. Supports wildcards in group names.


    Parameters
    ----------
    lh5_file
        name of file.
    lh5_group
        group to search. add a ``/`` to the end of the group name if you want to
        list all objects inside that group.
    """

    log.debug(
        f"Listing objects in '{lh5_file}'"
        + ("" if lh5_group == "" else f" (and group {lh5_group})")
    )

    lh5_st = LH5Store()
    # To use recursively, make lh5_file a h5group instead of a string
    if isinstance(lh5_file, str):
        lh5_file = lh5_st.gimme_file(lh5_file, "r")
        if lh5_group.startswith("/"):
            lh5_group = lh5_group[1:]

    if lh5_group == "":
        lh5_group = "*"

    splitpath = lh5_group.split("/", 1)
    matchingkeys = fnmatch.filter(lh5_file.keys(), splitpath[0])

    if len(splitpath) == 1:
        return matchingkeys
    else:
        ret = []
        for key in matchingkeys:
            ret.extend([f"{key}/{path}" for path in ls(lh5_file[key], splitpath[1])])
        return ret


def show(
    lh5_file: str | h5py.Group,
    lh5_group: str = "/",
    indent: str = "",
    header: bool = True,
) -> None:
    """Print a tree of LH5 file contents with LGDO datatype.

    Parameters
    ----------
    lh5_file
        the LH5 file.
    lh5_group
        print only contents of this HDF5 group.
    indent
        indent the diagram with this string.
    header
        print `lh5_group` at the top of the diagram.

    Examples
    --------
    >>> from pygama.lgdo import show
    >>> show("file.lh5", "/geds/raw")
    /geds/raw
    ├── channel · array<1>{real}
    ├── energy · array<1>{real}
    ├── timestamp · array<1>{real}
    ├── waveform · table{t0,dt,values}
    │   ├── dt · array<1>{real}
    │   ├── t0 · array<1>{real}
    │   └── values · array_of_equalsized_arrays<1,1>{real}
    └── wf_std · array<1>{real}
    """
    # open file
    if isinstance(lh5_file, str):
        lh5_file = h5py.File(expand_path(lh5_file), "r")

    # go to group
    if lh5_group != "/":
        lh5_file = lh5_file[lh5_group]

    if header:
        print(f"\033[1m{lh5_group}\033[0m")  # noqa: T201

    # get an iterator over the keys in the group
    it = iter(lh5_file)
    key = None

    # make sure there is actually something in this file/group
    try:
        key = next(it)  # get first key
    except StopIteration:
        print(f"{indent}└──  empty")  # noqa: T201
        return

    # loop over keys
    while True:
        val = lh5_file[key]
        # we want to print the LGDO datatype
        attr = val.attrs.get("datatype", default="no datatype")
        if attr == "no datatype" and isinstance(val, h5py.Group):
            attr = "HDF5 group"

        # is this the last key?
        killme = False
        try:
            k_new = next(it)  # get next key
        except StopIteration:
            char = "└──"
            killme = True  # we'll have to kill this loop later
        else:
            char = "├──"

        print(f"{indent}{char} \033[1m{key}\033[0m · {attr}")  # noqa: T201

        # if it's a group, call this function recursively
        if isinstance(val, h5py.Group):
            show(val, indent=indent + ("    " if killme else "│   "), header=False)

        # break or move to next key
        if killme:
            break
        else:
            key = k_new


def load_nda(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[np.ndarray | list | tuple] = None,
) -> dict[str, np.ndarray]:
    r"""Build a dictionary of :class:`numpy.ndarray`\ s from LH5 data.

    Given a list of files, a list of LH5 table parameters, and an optional
    group path, return a NumPy array with all values for each parameter.

    Parameters
    ----------
    f_list
        A list of files. Can contain wildcards.
    par_list
        A list of parameters to read from each file.
    lh5_group
        group path within which to find the specified parameters.
    idx_list
        for fancy-indexed reads. Must be one index array for each file in
        `f_list`.

    Returns
    -------
    par_data
        A dictionary of the parameter data keyed by the elements of `par_list`.
        Each entry contains the data for the specified parameter concatenated
        over all files in `f_list`.
    """
    if isinstance(f_list, str):
        f_list = [f_list]
        if idx_list is not None:
            idx_list = [idx_list]
    if idx_list is not None and len(f_list) != len(idx_list):
        raise ValueError(
            f"f_list length ({len(f_list)}) != idx_list length ({len(idx_list)})!"
        )

    # Expand wildcards
    f_list = [f for f_wc in f_list for f in sorted(glob.glob(os.path.expandvars(f_wc)))]

    sto = LH5Store()
    par_data = {par: [] for par in par_list}
    for ii, f in enumerate(f_list):
        f = sto.gimme_file(f, "r")
        for par in par_list:
            if f"{lh5_group}/{par}" not in f:
                raise RuntimeError(f"'{lh5_group}/{par}' not in file {f_list[ii]}")

            if idx_list is None:
                data, _ = sto.read_object(f"{lh5_group}/{par}", f)
            else:
                data, _ = sto.read_object(f"{lh5_group}/{par}", f, idx=idx_list[ii])
            if not data:
                continue
            par_data[par].append(data.nda)
    par_data = {par: np.concatenate(par_data[par]) for par in par_list}
    return par_data


def load_dfs(
    f_list: str | list[str],
    par_list: list[str],
    lh5_group: str = "",
    idx_list: list[np.ndarray | list | tuple] = None,
) -> pd.DataFrame:
    """Build a :class:`pandas.DataFrame` from LH5 data.

    Given a list of files (can use wildcards), a list of LH5 columns, and
    optionally the group path, return a :class:`pandas.DataFrame` with all
    values for each parameter.

    See Also
    --------
    :func:`load_nda`

    Returns
    -------
    dataframe
        contains columns for each parameter in `par_list`, and rows containing
        all data for the associated parameters concatenated over all files in
        `f_list`.
    """
    return pd.DataFrame(
        load_nda(f_list, par_list, lh5_group=lh5_group, idx_list=idx_list)
    )


def expand_path(path: str, list: bool = False) -> str | list:
    """Expand environment variables and wildcards to return absolute path

    Parameters
    ----------
    path
        name of path, which may include environment variables and wildcards
    list
        if True, return a list. If False, return a string; if False and a
        unique file is not found, raise an Exception

    Returns
    -------
    path or list of paths
        Unique absolute path, or list of all absolute paths
    """

    paths = glob.glob(os.path.expanduser(os.path.expandvars(path)))
    if not list:
        if len(paths) == 0:
            raise FileNotFoundError(f"could not find path matching {path}")
        elif len(paths) > 1:
            raise FileNotFoundError(f"found multiple paths matching {path}")
        else:
            return paths[0]
    else:
        return paths
