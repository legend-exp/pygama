"""
Implements a LEGEND Data Object representing an array of equal-sized arrays and
corresponding utilities.
"""
from __future__ import annotations

from typing import Any

import numpy

from pygama.lgdo.array import Array
from pygama.lgdo.lgdo_utils import get_element_type


class ArrayOfEqualSizedArrays(Array):
    """An array of equal-sized arrays.

    Arrays of equal size within a file but could be different from application
    to application. Canonical example: array of same-length waveforms.
    """

    def __init__(
        self,
        dims: tuple[int, ...] = None,
        nda: numpy.ndarray = None,
        shape: tuple[int, ...] = (),
        dtype: numpy.dtype = None,
        fill_val: int | float = None,
        attrs: dict[str, Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        dims
            specifies the dimensions required for building the
            :class:`ArrayOfEqualSizedArrays`' `datatype` attribute.
        nda
            An :class:`numpy.ndarray` to be used for this object's internal
            array. Note: the array is used directly, not copied. If not
            supplied, internal memory is newly allocated based on the `shape`
            and `dtype` arguments.
        shape
            A NumPy-format shape specification for shape of the internal
            array. Required if `nda` is ``None``, otherwise unused.
        dtype
            Specifies the type of the data in the array. Required if `nda` is
            ``None``, otherwise unused.
        fill_val
            If ``None``, memory is allocated without initialization. Otherwise,
            the array is allocated with all elements set to the corresponding
            fill value. If `nda` is not ``None``, this parameter is ignored.
        attrs
            A set of user attributes to be carried along with this LGDO.

        Notes
        -----
        If shape is not "1D array of arrays of shape given by axes 1-N" (of
        `nda`) then specify the dimensionality split in the constructor.

        See Also
        --------
        :class:`.Array`
        """
        self.dims = dims
        super().__init__(
            nda=nda, shape=shape, dtype=dtype, fill_val=fill_val, attrs=attrs
        )

    def datatype_name(self) -> str:
        return "array_of_equalsized_arrays"

    def form_datatype(self) -> str:
        dt = self.datatype_name()
        nd = str(len(self.nda.shape))
        if self.dims is not None:
            nd = ",".join([str(i) for i in self.dims])
        et = get_element_type(self)
        return dt + "<" + nd + ">{" + et + "}"

    def __len__(self) -> int:
        return len(self.nda)
