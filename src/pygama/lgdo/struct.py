"""
Implements a LEGEND Data Object representing a struct and corresponding
utilities.
"""

from __future__ import annotations

from typing import Any
import logging

from pygama.lgdo.array import Array
from pygama.lgdo.vectorofvectors import VectorOfVectors
from pygama.lgdo.scalar import Scalar

log = logging.getLogger(__name__)

LGDO = Array | VectorOfVectors | Scalar


class Struct(dict):
    """A dictionary of LGDO's with an optional set of attributes.

    After instantiation, add fields using :meth:`add_field` to keep the
    datatype updated, or call :meth:`update_datatype` after adding.
    """
    # TODO: overload setattr to require add_field for setting?

    def __init__(self, obj_dict: dict[str, LGDO | Struct] = {},
                 attrs: dict[str, Any] = {}) -> None:
        """
        Parameters
        ----------
        obj_dict
            instantiate this Struct using the supplied named LGDO's.  Note: no
            copy is performed, the objects are used directly.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        self.update(obj_dict)
        self.attrs = dict(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                log.warning((
                    "datatype does not match obj_dict! "
                    f"datatype: {self.attrs['datatype']}, "
                    f"obj_dict.keys(): {obj_dict.keys()}, "
                    f"form_datatype(): {self.form_datatype()}"))
                log.warning("will be updated to the latter.")
        self.update_datatype()

    def datatype_name(self) -> str:
        """The name for this LGDO's datatype attribute."""
        return 'struct'

    def form_datatype(self) -> str:
        """Return this LGDO's datatype attribute string."""
        return self.datatype_name() + '{' + ','.join(self.keys()) + '}'

    def update_datatype(self) -> None:
        self.attrs['datatype'] = self.form_datatype()

    def add_field(self, name: str, obj: LGDO | Struct) -> None:
        self[name] = obj
        self.update_datatype()

    def __len__(self) -> int:
        """As a convention for Structs, always return 1."""
        return 1

    def __str__(self) -> str:
        """Convert to string (e.g. for printing)."""
        tmp_attrs = self.attrs.copy()
        datatype = tmp_attrs.pop('datatype')
        # __repr__ instead of __str__ to avoid infinite loop
        string = datatype + " = " + super().__repr__()
        if len(tmp_attrs) > 0:
            string += '\nattrs = ' + str(tmp_attrs)
        return string

    def __repr__(self) -> str:
        return str(self)
