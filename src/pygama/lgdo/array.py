"""
Implements a LEGEND Data Object representing an n-dimensional array and
corresponding utilities.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pygama.lgdo.lgdo import LGDO
from pygama.lgdo.lgdo_utils import get_element_type

log = logging.getLogger(__name__)


class Array(LGDO):
    r"""Holds an :class:`numpy.ndarray` and attributes.

    :class:`Array` (and the other various array types) holds an `nda` instead
    of deriving from :class:`numpy.ndarray` for the following reasons:

    - It keeps management of the `nda` totally under the control of the user. The
      user can point it to another object's buffer, grab the `nda` and toss the
      :class:`Array`, etc.
    - It allows the management code to send just the `nda`'s the central routines
      for data manpulation. Keeping LGDO's out of that code allows for more
      standard, reusable, and (we expect) performant Python.
    - It allows the first axis of the `nda` to be treated as "special" for storage
      in :class:`.Table`\ s.
    """

    def __init__(
        self,
        nda: np.ndarray = None,
        shape: tuple[int, ...] = (),
        dtype: np.dtype = None,
        fill_val: float | int = None,
        attrs: dict[str, Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        nda
            An :class:`numpy.ndarray` to be used for this object's internal
            array. Note: the array is used directly, not copied. If not
            supplied, internal memory is newly allocated based on the shape and
            dtype arguments.
        shape
            A numpy-format shape specification for shape of the internal
            ndarray. Required if `nda` is ``None``, otherwise unused.
        dtype
            Specifies the type of the data in the array. Required if `nda` is
            ``None``, otherwise unused.
        fill_val
            If ``None``, memory is allocated without initialization. Otherwise,
            the array is allocated with all elements set to the corresponding
            fill value. If `nda` is not ``None``, this parameter is ignored.
        attrs
            A set of user attributes to be carried along with this LGDO.
        """
        if nda is None:
            if fill_val is None:
                nda = np.empty(shape, dtype=dtype)
            elif fill_val == 0:
                nda = np.zeros(shape, dtype=dtype)
            else:
                nda = np.full(shape, fill_val, dtype=dtype)
        self.nda = nda
        self.dtype = self.nda.dtype

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        dt = self.datatype_name()
        nd = str(len(self.nda.shape))
        et = get_element_type(self)
        return dt + "<" + nd + ">{" + et + "}"

    def __len__(self) -> int:
        return len(self.nda)

    def resize(self, new_size: int) -> None:
        """Resize the array to `new_size`."""
        new_shape = (new_size,) + self.nda.shape[1:]
        self.nda.resize(new_shape, refcheck=True)

    def __str__(self) -> str:
        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop("datatype")
        string = str(self.nda)
        if len(tmp_attrs) > 0:
            string += f" with attrs={tmp_attrs}"
        return string

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "("
            + np.array2string(self.nda, prefix=self.__class__.__name__ + " ")
            + f", attrs={repr(self.attrs)})"
        )
