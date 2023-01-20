"""
Implements a LEGEND Data Object representing an n-dimensional array of fixed
size and corresponding utilities.
"""
from __future__ import annotations

from typing import Any

import numpy

from pygama.lgdo.array import Array


class FixedSizeArray(Array):
    """An array of fixed-size arrays.

    Arrays with guaranteed shape along axes > 0: for example, an array of
    vectors will always length 3 on axis 1, and it will never change from
    application to application.  This data type is used for optimized memory
    handling on some platforms. We are not that sophisticated so we are just
    storing this identification for LGDO validity, i.e. for now this class is
    just an alias for :class:`~.Array`, but keeps track of the datatype name.
    """

    def __init__(
        self,
        nda: numpy.ndarray = None,
        shape: tuple[int, ...] = (),
        dtype: numpy.dtype = None,
        fill_val: int | float = None,
        attrs: dict[str, Any] = None,
    ) -> None:
        """
        See Also
        --------
        :class:`.Array`
        """
        super().__init__(
            nda=nda, shape=shape, dtype=dtype, fill_val=fill_val, attrs=attrs
        )

    def datatype_name(self) -> str:
        return "fixedsize_array"
