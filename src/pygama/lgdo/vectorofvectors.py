"""
Implements a LEGEND Data Object representing a variable-length array of
variable-length arrays and corresponding utilities.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numba import njit

from pygama.lgdo.array import Array
from pygama.lgdo.arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from pygama.lgdo.lgdo import LGDO
from pygama.lgdo.lgdo_utils import get_element_type

log = logging.getLogger(__name__)


class VectorOfVectors(LGDO):
    """A variable-length array of variable-length arrays.

    For now only a 1D vector of 1D vectors is supported. Internal representation
    is as two NumPy arrays, one to store the flattened data contiguosly and one
    to store the cumulative sum of lengths of each vector.
    """

    def __init__(
        self,
        flattened_data: Array = None,
        cumulative_length: Array = None,
        shape_guess: tuple[int, int] = None,
        dtype: np.dtype = None,
        attrs: dict[str, Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        flattened_data
            If not ``None``, used as the internal memory array for
            `flattened_data`.  Otherwise, an internal `flattened_data` is
            allocated based on `shape_guess` and `dtype`.
        cumulative_length
            If not ``None``, used as the internal memory array for
            `cumulative_length`. Should be `dtype` :any:`numpy.uint32`. If
            `cumulative_length` is ``None``, an internal `cumulative_length` is
            allocated based on the first element of `shape_guess`.
        shape_guess
            A NumPy-format shape specification, required if either of
            `flattened_data` or `cumulative_length` are not supplied.  The
            first element should not be a guess and sets the number of vectors
            to be stored. The second element is a guess or approximation of the
            typical length of a stored vector, used to set the initial length
            of `flattened_data` if it was not supplied.
        dtype
            Sets the type of data stored in `flattened_data`. Required if
            `flattened_data` is ``None``.
        attrs
            A set of user attributes to be carried along with this LGDO.
        """
        if cumulative_length is None:
            self.cumulative_length = Array(
                shape=(shape_guess[0],), dtype="uint32", fill_val=0
            )
        else:
            if isinstance(cumulative_length, Array):
                self.cumulative_length = cumulative_length
            else:
                self.cumulative_length = Array(cumulative_length)
        if flattened_data is None:
            length = np.prod(shape_guess)
            if dtype is None:
                raise ValueError("flattened_data and dtype cannot both be None!")
            else:
                self.flattened_data = Array(shape=(length,), dtype=dtype)
                self.dtype = np.dtype(dtype)
        else:
            if isinstance(flattened_data, Array):
                self.flattened_data = flattened_data
            else:
                self.flattened_data = Array(flattened_data)
            if dtype is None:
                self.dtype = self.flattened_data.dtype
            else:
                self.dtype = np.dtype(dtype)

        self.attrs = {} if attrs is None else dict(attrs)

        super().__init__()

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        et = get_element_type(self)
        return "array<1>{array<1>{" + et + "}}"

    def __len__(self) -> int:
        """Provides ``__len__`` for this array-like class."""
        return len(self.cumulative_length)

    def resize(self, new_size: int) -> None:
        self.cumulative_length.resize(new_size)

    def set_vector(self, i_vec: int, nda: np.ndarray) -> None:
        """Insert vector `nda` at location `i_vec`.

        Notes
        -----
        `flattened_data` is doubled in length until `nda` can be appended to
        it.
        """
        if i_vec < 0 or i_vec > len(self.cumulative_length.nda) - 1:
            raise ValueError("bad i_vec", i_vec)

        if len(nda.shape) != 1:
            raise ValueError("nda had bad shape", nda.shape)

        start = 0 if i_vec == 0 else self.cumulative_length.nda[i_vec - 1]
        end = start + len(nda)
        while end >= len(self.flattened_data.nda):
            self.flattened_data.nda.resize(
                2 * len(self.flattened_data.nda), refcheck=True
            )
        self.flattened_data.nda[start:end] = nda
        self.cumulative_length.nda[i_vec] = end

    def get_vector(self, i_vec: int) -> np.ndarray:
        """Get vector at index `i_vec`."""
        if i_vec >= len(self.cumulative_length) or i_vec < 0:
            raise IndexError

        if i_vec == 0:
            start = 0
            end = self.cumulative_length.nda[0]
        else:
            start = self.cumulative_length.nda[i_vec - 1]
            end = self.cumulative_length.nda[i_vec]

        return self.flattened_data.nda[start:end]

    def __iter__(self) -> VectorOfVectors:
        self.index = 0
        return self

    def __next__(self) -> np.ndarray:
        try:
            if self.index == 0:
                start = 0
                end = self.cumulative_length.nda[0]
            else:
                start = self.cumulative_length.nda[self.index - 1]
                end = self.cumulative_length.nda[self.index]
            result = self.flattened_data.nda[start:end]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

    def __getitem__(self, index: int) -> list:
        return list(self)[index]

    def __str__(self) -> str:
        string = ""
        pos = 0
        for vec in self:
            if pos != 0:
                string += " "

            string += np.array2string(vec, prefix=" ")

            if pos < len(self.cumulative_length):
                string += ",\n"

            pos += 1

        string = f"[{string}]"

        tmp_attrs = self.attrs.copy()
        tmp_attrs.pop("datatype")
        if len(tmp_attrs) > 0:
            string += f" with attrs={tmp_attrs}"

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(threshold=5, edgeitems=2, linewidth=100)
        out = (
            "VectorOfVectors(flattened_data="
            + repr(self.flattened_data)
            + ", cumulative_length="
            + repr(self.cumulative_length)
            + ", attrs="
            + repr(self.attrs)
            + ")"
        )
        np.set_printoptions(**npopt)
        return out

    def to_aoesa(self) -> ArrayOfEqualSizedArrays:
        """Convert to ArrayOfEqualSizedArrays, padding with NaNs"""
        ind_lengths = np.diff(self.cumulative_length.nda, prepend=0)
        arr_len = np.max(ind_lengths)
        nda = np.empty((len(self.cumulative_length), arr_len))
        nda.fill(np.nan)
        for i in range(len(self.cumulative_length)):
            nda[i, : ind_lengths[i]] = self[i]
        return ArrayOfEqualSizedArrays(nda=nda)


def build_cl(
    sorted_array_in: Array, cumulative_length_out: np.ndarray = None
) -> np.ndarray:
    """build a cumulative_length array from an array of sorted data

    So for example if sorted_array_in contains [ 3, 3, 3, 4 ], would return
    [ 2, 3 ]

    For a sorted_array_in of indices, this is the inverse of explode_cl() below,
    in the sense that doing build_cl(explode_cl(cumulative_length)) would
    recover the original cumulative_length.

    Parameters
    ----------
    sorted_array_in
        Array of data already sorted; each N matching contiguous entries will be
        converted into a new row of cumulative_length_out
    cumulative_length_out
        This is an optional pre-allocated array for the output
        cumulative_length. It will always have length <= sorted_array_in, so
        giving them the same length is safe if there is not a better guess.

    Returns
    -------
    cumulative_length_out
        The output cumulative_length. If the user provides a
        cumulative_length_out that is too long, this return value is sliced to
        contain only the used portion of the allocated memory
    """
    if len(sorted_array_in) == 0:
        return None
    sorted_array_in = np.asarray(sorted_array_in)
    if cumulative_length_out is None:
        cumulative_length_out = np.zeros(len(sorted_array_in), dtype=np.uint64)
    else:
        cumulative_length_out.fill(0)
    if len(cumulative_length_out) == 0 and len(sorted_array_in) > 0:
        raise ValueError(
            "cumulative_length_out too short ({len(cumulative_length_out)})"
        )
    return nb_build_cl(sorted_array_in, cumulative_length_out)


@njit
def nb_build_cl(
    sorted_array_in: np.ndarray, cumulative_length_out: np.ndarray
) -> np.ndarray:
    """numbified inner loop for build_cl"""
    ii = 0
    last_val = sorted_array_in[0]
    for val in sorted_array_in:
        if val != last_val:
            ii += 1
            cumulative_length_out[ii] = cumulative_length_out[ii - 1]
            if ii >= len(cumulative_length_out):
                raise RuntimeError("cumulative_length_out too short")
            last_val = val
        cumulative_length_out[ii] += 1
    ii += 1
    return cumulative_length_out[:ii]


def explode_cl(cumulative_length: Array, array_out: np.ndarray = None) -> np.ndarray:
    """explode a cumulative_length array

    So for example if cumulative_length is [ 2, 3 ], would return [ 0, 0, 0, 1]

    This is the inverse of build_cl() above, in the sense that doing
    build_cl(explode_cl(cumulative_length)) would recover the original
    cumulative_length.

    Parameters
    ----------
    cumulative_length
        the cumulative_length array to be exploded
    array_out
        an optional pre-allocated array to hold the exploded cumulative_length.
        The length should be equal to cumulative_length[-1]

    Returns
    -------
    array_out
        the exploded cumulative_length array
    """
    cumulative_length = np.asarray(cumulative_length)
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if array_out is None:
        array_out = np.empty(int(out_len), dtype=np.uint64)
    if len(array_out) != out_len:
        raise ValueError(
            f"bad lengths: cl[-1] ({cumulative_length[-1]}) != out ({len(array_out)})"
        )
    return nb_explode_cl(cumulative_length, array_out)


@njit
def nb_explode_cl(cumulative_length: np.ndarray, array_out: np.ndarray) -> np.ndarray:
    """numbified inner loop for explode_cl"""
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if len(array_out) != out_len:
        raise ValueError("bad lengths")
    start = 0
    for ii in range(len(cumulative_length)):
        nn = int(cumulative_length[ii] - start)
        for jj in range(nn):
            array_out[int(start + jj)] = ii
        start = cumulative_length[ii]
    return array_out


def explode(
    cumulative_length: Array, array_in: Array, array_out: np.ndarray = None
) -> np.ndarray:
    """explode a data array using a cumulative_length array

    This is identical to allocated_explode_cl, except array_in gets exploded
    instead of cumulative_length. So for example, if array_in = [ 3, 4 ] and
    cumulative_length = [ 2, 3 ], array_out would be [ 3, 3, 3, 4 ]

    Parameters
    ----------
    cumulative_length
        the cumulative_length array to use for exploding
    array_in
        the data to be exploded. Must have same length as cumulative_length
    array_out
        a pre-allocated array to hold the exploded data. The length should be
        equal to cumulative_length[-1]
    """
    cumulative_length = np.asarray(cumulative_length)
    array_in = np.asarray(array_in)
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if array_out is None:
        array_out = np.empty(out_len, dtype=array_in.dtype)
    if len(cumulative_length) != len(array_in) or len(array_out) != out_len:
        raise ValueError(
            f"bad lengths: cl ({len(cumulative_length)}) != in ({len(array_in)}) and cl[-1] ({cumulative_length[-1]}) != out ({len(array_out)})"
        )
    return nb_explode(cumulative_length, array_in, array_out)


@njit
def nb_explode(
    cumulative_length: np.ndarray, array_in: np.ndarray, array_out: np.ndarray
) -> np.ndarray:
    """numbified inner loop for explode"""
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if len(cumulative_length) != len(array_in) or len(array_out) != out_len:
        raise ValueError("bad lengths")
    ii = 0
    for jj in range(len(array_out)):
        while ii < len(cumulative_length) and jj >= cumulative_length[ii]:
            ii += 1
        array_out[jj] = array_in[ii]
    return array_out


def explode_arrays(
    cumulative_length: Array, arrays: list, out_arrays: list = None
) -> list:
    """explode a set of arrays using a cumulative_length array

    Parameters
    ----------
    cumulative_length
        the cumulative_length array to use for exploding
    arrays
        the data arrays to be exploded. Each array must have same length as
        cumulative_length
    out_arrays
        an optional list of pre-allocated arrays to hold the exploded data. The
        length of the list should be equal to the number of "arrays", and each
        entry in array_out should have length cumulative_length[-1]. If not
        provided, output arrays are allocated for the user.
    """
    cumulative_length = np.asarray(cumulative_length)
    for ii in range(len(arrays)):
        arrays[ii] = np.asarray(arrays[ii])
    out_len = cumulative_length[-1] if len(cumulative_length) > 0 else 0
    if out_arrays is None:
        out_arrays = []
        for array in arrays:
            out_arrays.append(np.empty(out_len, dtype=array.dtype))
    for ii in range(len(arrays)):
        explode(cumulative_length, arrays[ii], out_arrays[ii])
    return out_arrays
