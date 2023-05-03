"""Implements utilities for LEGEND Data Objects."""
from __future__ import annotations

import glob
import logging
import os
import string

import numpy as np

from .. import lgdo

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


def copy(obj: lgdo.LGDO, dtype: np.dtype = None) -> lgdo.LGDO:
    """Return a copy of an LGDO.

    Parameters
    ----------
    obj
        the LGDO to be copied.
    dtype
        NumPy dtype to be used for the copied object.

    """
    if dtype is None:
        dtype = obj.dtype

    if isinstance(obj, lgdo.Array):
        return lgdo.Array(
            np.array(obj.nda, dtype=dtype, copy=True), attrs=dict(obj.attrs)
        )

    if isinstance(obj, lgdo.VectorOfVectors):
        return lgdo.VectorOfVectors(
            flattened_data=copy(obj.flattened_data, dtype=dtype),
            cumulative_length=copy(obj.cumulative_length),
            attrs=dict(obj.attrs),
        )

    else:
        raise ValueError(f"copy of {type(obj)} not supported")


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


def expand_vars(expr: str, substitute: dict[str, str] = None) -> str:
    """Expand (environment) variables.

    Note
    ----
    Malformed variable names and references to non-existing variables are left
    unchanged.

    Parameters
    ----------
    expr
        string expression, which may include (environment) variables prefixed by
        ``$``.
    substitute
        use this dictionary to substitute variables. Environment variables take
        precedence.
    """
    if substitute is None:
        substitute = {}

    # expand env variables first
    # then try using provided mapping
    return string.Template(os.path.expandvars(expr)).safe_substitute(substitute)


def expand_path(
    path: str,
    substitute: dict[str, str] = None,
    list: bool = False,
    base_path: str = None,
) -> str | list:
    """Expand (environment) variables and wildcards to return absolute paths.

    Parameters
    ----------
    path
        name of path, which may include environment variables and wildcards.
    list
        if ``True``, return a list. If ``False``, return a string; if ``False``
        and a unique file is not found, raise an exception.
    substitute
        use this dictionary to substitute variables. Environment variables take
        precedence.
    base_path
        name of base path. Returned paths will be relative to base.

    Returns
    -------
    path or list of paths
        Unique absolute path, or list of all absolute paths
    """
    if base_path is not None and base_path != "":
        base_path = os.path.expanduser(os.path.expandvars(base_path))
        path = os.path.join(base_path, path)

    # first expand variables
    _path = expand_vars(path, substitute)

    # then expand wildcards
    paths = glob.glob(os.path.expanduser(_path))

    if base_path is not None and base_path != "":
        paths = [os.path.relpath(p, base_path) for p in paths]

    if not list:
        if len(paths) == 0:
            raise FileNotFoundError(f"could not find path matching {path}")
        elif len(paths) > 1:
            raise FileNotFoundError(f"found multiple paths matching {path}")
        else:
            return paths[0]
    else:
        return paths
