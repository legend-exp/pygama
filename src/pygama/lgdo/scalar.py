"""
Implements a LEGEND Data Object representing a scalar and corresponding
utilities.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pygama.lgdo.lgdo_utils import get_element_type

log = logging.getLogger(__name__)


class Scalar:
    """Holds just a scalar value and some attributes (datatype, units, ...).
    """
    # TODO: do scalars need proper numpy dtypes?

    def __init__(self, value: int | float,
                 attrs: dict[str, Any] = {}) -> None:
        """
        Parameters
        ----------
        value
            the value for this scalar.
        attrs
            a set of user attributes to be carried along with this LGDO.
        """
        if not np.isscalar(value):
            raise ValueError('cannot instantiate a Scalar with a non-scalar value')

        self.value = value
        self.attrs = dict(attrs)
        if 'datatype' in self.attrs:
            if self.attrs['datatype'] != self.form_datatype():
                log.warning(
                    f"datatype ({self.attrs['datatype']}) does "
                    f"not match value type ({type(value).__name__})!")
        else:
            self.attrs['datatype'] = get_element_type(self.value)

    def datatype_name(self) -> str:
        """The name for this LGDO's datatype attribute."""
        if hasattr(self.value, 'datatype_name'):
            return self.value.datatype_name
        else:
            return get_element_type(self.value)

    def form_datatype(self) -> str:
        """Return this LGDO's datatype attribute string."""
        return self.datatype_name()

    def __str__(self) -> str:
        """Convert to string (e.g. for printing)."""
        string = str(self.value)
        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop('datatype')
        if len(tmp_attrs) > 0:
            string += '\n' + str(tmp_attrs)
        return string

    def __repr__(self) -> str:
        return str(self)
