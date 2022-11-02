"""
Implements utilities for LEGEND Data Objects.
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np

log = logging.getLogger(__name__)


def get_element_type(obj: object) -> str:
    """Get the LGDO element type of a scalar or array.

    For use in LGDO datatype attributes.

    Parameters
    ----------
    obj
        if a ``str``, will automatically return ``string`` if the object has
        a :class:`numpy.dtype`, that will be used for determining the element
        type otherwise will attempt to case the type of the object to a
        :class:`numpy.dtype`.

    Returns
    -------
    element_type
        A string stating the determined element type of the object.
    """

    # special handling for strings
    if isinstance(obj, str):
        return "string"

    # the rest use dtypes
    dt = obj.dtype if hasattr(obj, "dtype") else np.dtype(type(obj))
    kind = dt.kind

    if kind == "b":
        return "bool"
    if kind == "V":
        return "blob"
    if kind in ["i", "u", "f"]:
        return "real"
    if kind == "c":
        return "complex"
    if kind in ["S", "U"]:
        return "string"

    # couldn't figure it out
    raise ValueError(
        "cannot determine lgdo element_type for object of type", type(obj).__name__
    )


def parse_datatype(datatype: str) -> tuple[str, tuple[int, ...], str | list[str]]:
    """Parse datatype string and return type, dimensions and elements.

    Parameters
    ----------
    datatype
        a LGDO-formatted datatype string.

    Returns
    -------
    element_type
        the datatype name dims if not ``None``, a tuple of dimensions for the
        LGDO. Note this is not the same as the NumPy shape of the underlying
        data object. See the LGDO specification for more information. Also see
        :class:`.ArrayOfEqualSizedArrays` and
        :meth:`.lh5_store.LH5Store.read_object` for example code elements for
        numeric objects, the element type for struct-like  objects, the list of
        fields in the struct.
    """
    if "{" not in datatype:
        return "scalar", None, datatype

    # for other datatypes, need to parse the datatype string
    from parse import parse

    datatype, element_description = parse("{}{{{}}}", datatype)
    if datatype.endswith(">"):
        datatype, dims = parse("{}<{}>", datatype)
        dims = [int(i) for i in dims.split(",")]
        return datatype, tuple(dims), element_description
    else:
        return datatype, None, element_description.split(",")


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
