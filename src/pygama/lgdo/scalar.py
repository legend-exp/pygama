"""Implements a LEGEND Data Object representing a scalar and corresponding utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pygama.lgdo.lgdo_utils import get_element_type

log = logging.getLogger(__name__)


class Scalar:
    """Holds just a scalar value and some attributes (datatype, units, ...)."""

    # TODO: do scalars need proper numpy dtypes?

    def __init__(self, value: int | float, attrs: dict[str, Any] = None) -> None:
        """
        Parameters
        ----------
        value
            the value for this scalar.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        if not np.isscalar(value):
            raise ValueError("cannot instantiate a Scalar with a non-scalar value")

        self.value = value
        self.attrs = {} if attrs is None else dict(attrs)

        if "datatype" in self.attrs:
            if self.attrs["datatype"] != self.form_datatype():
                raise ValueError(
                    f"datatype ({self.attrs['datatype']}) does "
                    f"not match value type ({type(value).__name__})!"
                )
        else:
            self.attrs["datatype"] = get_element_type(self.value)

    def datatype_name(self) -> str:
        """Returns the name for this LGDO's datatype attribute."""
        if hasattr(self.value, "datatype_name"):
            return self.value.datatype_name
        else:
            return get_element_type(self.value)

    def form_datatype(self) -> str:
        """Return this LGDO's datatype attribute string."""
        return self.datatype_name()

    def __str__(self) -> str:
        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop("datatype")
        return f"{str(self.value)} with attrs={repr(tmp_attrs)}"

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(value={repr(self.value)}, attrs={repr(self.attrs)})"
        )
