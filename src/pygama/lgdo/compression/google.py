"""Google's data encoding algorithms.

Full specification at <https://protobuf.dev/programming-guides/encoding>.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
from numpy import ubyte
from numpy.typing import NDArray

from .base import WaveformCodec


@dataclass(frozen=True)
class GoogleProtobuf(WaveformCodec):
    """Google's Protocol Buffers variable-width integer (*varint*) encoding `[ref] <https://protobuf.dev/programming-guides/encoding>`_."""

    zigzag: bool = True
    """Enable ZigZag encoding."""


@numba.jit
def zigzag_encode(x: int | NDArray[int]) -> int | NDArray[int]:
    """ZigZag-encode integer numbers."""
    return (x >> 31) ^ (x << 1)


@numba.jit
def zigzag_decode(x: int | NDArray[int]) -> int | NDArray[int]:
    """ZigZag-decode integer numbers."""
    return (x >> 1) ^ -(x & 1)


@numba.jit
def unsigned_varint_encode(x: int, encx: NDArray[ubyte]) -> int:
    """Compute the varint representation of an unsigned integer number.

    Only positive numbers are expected, as no *two’s complement* is applied.

    Parameters
    ----------
    x
        the number to be encoded.
    encx
        the encoded varint as a NumPy array of bytes.

    Returns
    -------
    nbytes
        size of varint in bytes
    """
    i = 0
    bits = x & 0x7F
    x >>= 7
    while x:
        encx[i] = 0x80 | bits
        bits = x & 0x7F
        i += 1
        x >>= 7

    encx[i] = bits
    # return size of varint in bytes
    return i + 1


@numba.jit
def unsigned_varint_decode(encx: NDArray[ubyte]) -> (int, int):
    """Decode a varint into an unsigned integer number.

    Only encoded positive numbers are expected, as no *two’s complement* is
    applied.

    Parameters
    ----------
    encx
        the encoded varint as a NumPy array of bytes.

    Returns
    -------
    x, nread
        the decoded value and the number of bytes read from the input array.
    """
    x = 0
    pos = 0
    for b in encx:
        x = x | ((b & 0x7F) << pos)
        if (b & 0x80) == 0:
            return (x, int(pos / 7 + 1))
        else:
            pos += 7
        if pos >= 64:
            raise RuntimeError("overflow during decoding of varint encoded number.")


@numba.guvectorize(
    [
        "void(uint64[:], uint8[:])",
        "void(uint32[:], uint8[:])",
        "void(uint16[:], uint8[:])",
    ],
    "(n),(m)",
)
def unsigned_varint_array_encode(
    sig_in: NDArray[int],
    sig_out: NDArray[ubyte],
) -> None:
    """Encode an array of unsigned integer numbers.

    Stores the number of bytes written in the first element of `sig_out`. The
    actual encoded data is found in ``sig_out[1:sig_out[0]]``.

    Parameters
    ----------
    sig_in
        the input array of integers.
    sig_out
        pre-allocated array for the output encoded data.

    See Also
    --------
    .unsigned_varint_encode, .unsigned_varint_array_decode
    """
    pos = 1
    for s in sig_in:
        pos += unsigned_varint_encode(s, sig_out[pos:])
    # store total number of bytes in the encoded array at the start of the same
    # array
    sig_out[0] = pos - 1


@numba.guvectorize(
    [
        "void(uint8[:], uint64[:])",
        "void(uint8[:], uint32[:])",
        "void(uint8[:], uint16[:])",
    ],
    "(n),(m)",
)
def unsigned_varint_array_decode(sig_in: NDArray[ubyte], sig_out: NDArray[int]) -> None:
    """Decode an array of varints, as returned by :func:`.unsigned_varint_array_encode`.

    Expects to find the actual number of bytes of the encoded signal in the
    first element of `sig_int`.

    Parameters
    ----------
    sig_in
        the array of varints.
    sig_out
        pre-allocated array for the output decoded integers.

    See Also
    --------
    .unsigned_varint_decode, .unsigned_varint_array_encode
    """
    pos = 1
    i = 0
    while pos <= sig_in[0]:
        x, nread = unsigned_varint_decode(sig_in[pos:])
        pos += nread
        sig_out[i] = x
        i += 1
