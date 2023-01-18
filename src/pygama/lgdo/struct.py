"""
Implements a LEGEND Data Object representing a struct and corresponding
utilities.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pygama.lgdo import LGDO

log = logging.getLogger(__name__)


class Struct(LGDO, dict):
    """A dictionary of LGDO's with an optional set of attributes.

    After instantiation, add fields using :meth:`add_field` to keep the
    datatype updated, or call :meth:`update_datatype` after adding.
    """

    # TODO: overload setattr to require add_field for setting?

    def __init__(
        self, obj_dict: dict[str, LGDO] = None, attrs: dict[str, Any] = None
    ) -> None:
        """
        Parameters
        ----------
        obj_dict
            instantiate this Struct using the supplied named LGDO's.  Note: no
            copy is performed, the objects are used directly.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        if obj_dict is not None:
            self.update(obj_dict)

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "struct"

    def form_datatype(self) -> str:
        return self.datatype_name() + "{" + ",".join(self.keys()) + "}"

    def update_datatype(self) -> None:
        self.attrs["datatype"] = self.form_datatype()

    def add_field(self, name: str, obj: LGDO) -> None:
        """Add a field to the table."""
        self[name] = obj
        self.update_datatype()

    def remove_field(self, name: str, delete: bool = False) -> None:
        """Remove a field from the table.

        Parameters
        ----------
        name
            name of the field to be removed
        delete
            if ``True``, delete the field object by calling :any:`del`.
        """
        if delete:
            del self[name]
        else:
            self.pop(name)
        self.update_datatype()

    def __str__(self) -> str:
        """Convert to string (e.g. for printing)."""

        thr_orig = np.get_printoptions()["threshold"]
        np.set_printoptions(threshold=8)

        string = "{\n"
        for k, v in self.items():
            if "\n" in str(v):
                rv = str(v).replace("\n", "\n    ")
                string += f" '{k}':\n    {rv},\n"
            else:
                string += f" '{k}': {v},\n"
        string += "}"

        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop("datatype")
        if tmp_attrs:
            string += f" with attrs={tmp_attrs}"

        np.set_printoptions(threshold=thr_orig)

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(threshold=5, edgeitems=2, linewidth=100)
        out = (
            self.__class__.__name__
            + "(dict="
            + dict.__repr__(self)
            + f", attrs={repr(self.attrs)})"
        )
        np.set_printoptions(**npopt)
        return " ".join(out.replace("\n", " ").split())
