from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .array import Array
from .lgdo import LGDO
from .lgdo_utils import get_element_type
from .vectorofvectors import VectorOfVectors


class VectorOfEncodedVectors(LGDO):
    """A variable-length array of variable-length encoded arrays.

    Mainly used to represent a vector of vector (lossless) encodings. In
    addition to :class:`~.vectorofvectors.VectorOfVectors`, a 1D
    :class:`~.array.Array` is stored in the ``encoded_size`` class attribute to
    hold the original size of the encoded vectors.

    See Also
    --------
    .vectorofvectors.VectorOfVectors
    """

    def __init__(
        self,
        encoded_data: VectorOfVectors = None,
        decoded_size: Array = None,
        attrs: dict[str, Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        encoded_data
            the vector of encoded vectors.
        decoded_size
            an array holding the original length of each encoded vector in
            `encoded_data`.
        attrs
            A set of user attributes to be carried along with this LGDO. Should
            include information about the codec used to encode the data.
        """
        if isinstance(encoded_data, VectorOfVectors):
            self.encoded_data = encoded_data
        else:
            raise ValueError("encoded_data must be a valid VectorOfVectors")

        if isinstance(decoded_size, Array):
            self.decoded_size = decoded_size
        elif decoded_size is not None:
            self.decoded_size = Array(decoded_size)
        elif encoded_data is not None:
            self.decoded_size = Array(
                shape=len(encoded_data), dtype="uint32", fill_val=0
            )

        if len(self.encoded_data) != len(self.decoded_size):
            raise RuntimeError("encoded_data vs. decoded_size shape mismatch")

        super().__init__(attrs)

    def datatype_name(self) -> str:
        return "array"

    def form_datatype(self) -> str:
        et = get_element_type(self.encoded_data)
        return "array<1>{encoded_array<1>{" + et + "}}"

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __eq__(self, other: VectorOfEncodedVectors) -> bool:
        if isinstance(other, VectorOfEncodedVectors):
            return (
                self.encoded_data == other.encoded_data
                and self.decoded_size == other.decoded_size
                and self.attrs == other.attrs
            )

        else:
            return False

    def resize(self, new_size: int) -> None:
        self.encoded_data.resize(new_size)
        self.decoded_size.resize(new_size)

    def set_vector(self, i_vec: int, nda: np.ndarray, dec_size: int) -> None:
        """Insert encoded vector `nda` and corresponding decoded size at
        location `i_vec`.

        See Also
        --------
        ~.vectorofvectors.VectorOfVectors

        Notes
        -----
        `self.decoded_size` is doubled in length until `dec_size` can be
        appended to it.
        """
        self.encoded_data.set_vector(i_vec, nda)

        while i_vec > len(self.decoded_size) - 1:
            self.decoded_size.resize(2 * len(self.decoded_size))
        self.decoded_size[i_vec] = dec_size

    def __setitem__(self, i_vec: int, value: tuple[np.ndarray, int]) -> None:
        return self.set_vector(i_vec, value[0], value[1])

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        """Return vector at index `i`."""
        return (self.encoded_data[i], self.decoded_size[i])

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        yield from zip(self.encoded_data, self.decoded_size)

    def __str__(self) -> str:
        string = ""
        pos = 0
        for vec, size in self:
            if pos != 0:
                string += " "

            string += (
                np.array2string(
                    vec,
                    prefix=" ",
                    formatter={
                        "int": lambda x, vec=vec: f"0x{x:02x}"
                        if vec.dtype == np.ubyte
                        else str(x)
                    },
                )
                + f" decoded_size = {size}"
            )

            if pos < len(self.encoded_data.cumulative_length):
                string += ",\n"

            pos += 1

        string = f"[{string}]"

        attrs = self.getattrs()
        if len(attrs) > 0:
            string += f" with attrs={attrs}"

        return string

    def __repr__(self) -> str:
        npopt = np.get_printoptions()
        np.set_printoptions(
            threshold=5,
            edgeitems=2,
            linewidth=100,
        )
        out = (
            "VectorOfEncodedVectors(encoded_data="
            + repr(self.encoded_data)
            + ", decoded_size="
            + repr(self.decoded_size)
            + ", attrs="
            + repr(self.attrs)
            + ")"
        )
        np.set_printoptions(**npopt)
        return out
