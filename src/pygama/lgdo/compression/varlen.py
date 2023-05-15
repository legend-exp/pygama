"""Variable-length code compression algorithms."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numba
import numpy as np
from numpy import int32, ubyte, uint32
from numpy.typing import NDArray

from pygama import lgdo

from .base import WaveformCodec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ULEB128ZigZagDiff(WaveformCodec):
    """ZigZag [#WikiZZ]_ encoding followed by Unsigned Little Endian Base 128 (ULEB128) [#WikiULEB128]_ encoding of array differences.

    .. [#WikiZZ] https://wikipedia.org/wiki/Variable-length_quantity#Zigzag_encoding
    .. [#WikiULEB128] https://wikipedia.org/wiki/LEB128#Unsigned_LEB128
    """

    codec: str = "uleb128_zigzag_diff"


def encode(
    sig_in: NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays,
    sig_out: NDArray[ubyte] = None,
) -> (NDArray[ubyte], NDArray[uint32]) | lgdo.VectorOfEncodedVectors:
    """Compress digital signal(s) with a variable-length encoding of its derivative.

    Wraps :func:`uleb128_zigzag_diff_array_encode` and adds support for encoding
    LGDOs.

    Note
    ----
    If `sig_in` is a NumPy array, no resizing of `sig_out` is performed. Not
    even of the internally allocated one.

    Because of the current implementation, providing a pre-allocated
    :class:`.VectorOfEncodedVectors` as `sig_out` is not possible.

    Parameters
    ----------
    sig_in
        array(s) holding the input signal(s).
    sig_out
        pre-allocated unsigned 8-bit integer array(s) for the compressed
        signal(s). If not provided, a new one will be allocated.

    Returns
    -------
    sig_out, nbytes
        given pre-allocated `sig_out` structure or new structure of unsigned
        8-bit integers, plus the number of bytes (length) of the encoded
        signal. If `sig_in` is an :class:`.LGDO`, only a newly allocated
        :class:`.VectorOfEncodedVectors` is returned.

    See Also
    --------
    uleb128_zigzag_diff_array_encode
    """
    if isinstance(sig_in, np.ndarray):
        s = sig_in.shape
        if len(sig_in) == 0:
            return np.empty(s[:-1] + (0,), dtype=ubyte), np.empty(0, dtype=uint32)

        if sig_out is None:
            # the encoded signal is an array of bytes
            # pre-allocate ubyte (uint8) array with a generous (but safe) size
            max_b = int(np.ceil(np.iinfo(sig_in.dtype).bits / 16) * 5)
            # expand last dimension
            sig_out = np.empty(s[:-1] + (s[-1] * max_b,), dtype=ubyte)

        if sig_out.dtype != ubyte:
            raise ValueError("sig_out must be of type ubyte")

        # nbytes has one dimension less (the last one)
        nbytes = np.empty(s[:-1], dtype=uint32)

        uleb128_zigzag_diff_array_encode(sig_in, sig_out, nbytes)

        # return without resizing
        return sig_out, nbytes

    elif isinstance(sig_in, lgdo.VectorOfVectors):
        if sig_out:
            log.warning(
                "a pre-allocated VectorOfEncodedVectors was given "
                "to hold an encoded ArrayOfEqualSizedArrays. "
                "This is not supported at the moment, so a new one "
                "will be allocated to replace it"
            )
        # convert VectorOfVectors to ArrayOfEqualSizedArrays so it can be
        # directly passed to the low-level encoding routine
        sig_out_nda, nbytes = encode(sig_in.to_aoesa())

        # build the encoded LGDO
        encoded_data = lgdo.ArrayOfEqualSizedArrays(nda=sig_out_nda).to_vov(
            cumulative_length=np.cumsum(nbytes, dtype=uint32)
        )
        # decoded_size is an array, compute it by diff'ing the original VOV
        decoded_size = np.diff(sig_in.cumulative_length, prepend=uint32(0))

        sig_out = lgdo.VectorOfEncodedVectors(encoded_data, decoded_size)

        return sig_out

    elif isinstance(sig_in, lgdo.ArrayOfEqualSizedArrays):
        if sig_out:
            log.warning(
                "a pre-allocated VectorOfEncodedVectors was given "
                "to hold an encoded ArrayOfEqualSizedArrays. "
                "This is not supported at the moment, so a new one "
                "will be allocated to replace it"
            )

        # encode the internal numpy array
        sig_out_nda, nbytes = encode(sig_in.nda)

        # build the encoded LGDO
        encoded_data = lgdo.ArrayOfEqualSizedArrays(nda=sig_out_nda).to_vov(
            cumulative_length=np.cumsum(nbytes, dtype=uint32)
        )
        sig_out = lgdo.ArrayOfEncodedEqualSizedArrays(
            encoded_data, decoded_size=sig_in.nda.shape[1]
        )

        return sig_out

    elif isinstance(sig_in, lgdo.Array):
        # encode the internal numpy array
        sig_out_nda, nbytes = encode(sig_in.nda, sig_out)
        return lgdo.Array(sig_out_nda), nbytes

    else:
        raise ValueError(f"unsupported input signal type ({type(sig_in)})")


def decode(
    sig_in: (NDArray[ubyte], NDArray[uint32]) | lgdo.VectorOfEncodedVectors,
    sig_out: NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays = None,
) -> NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays:
    """Deompress digital signal(s) with a variable-length encoding of its derivative.

    Wraps :func:`uleb128_zigzag_diff_array_decode` and adds support for decoding
    LGDOs.

    Note
    ----
    If `sig_in` is a NumPy array, no resizing (along the last dimension) of
    `sig_out` to its actual length is performed. Not even of the internally
    allocated one. If a pre-allocated :class:`.ArrayOfEqualSizedArrays` is
    provided, it won't be resized too. The internally allocated
    :class:`.ArrayOfEqualSizedArrays` `sig_out` has instead always the correct
    size.

    Because of the current implementation, providing a pre-allocated
    :class:`.VectorOfVectors` as `sig_out` is not possible.

    Parameters
    ----------
    sig_in
        array(s) holding the input, compressed signal(s). Output of
        :func:`.encode`.
    sig_out
        pre-allocated array(s) for the decompressed signal(s).  If not
        provided, will allocate a 32-bit integer array(s) structure.

    Returns
    -------
    sig_out
        given pre-allocated structure or new structure of 32-bit integers.

    See Also
    --------
    uleb128_zigzag_diff_array_decode
    """
    # expect the output of encode()
    if isinstance(sig_in, tuple):
        if sig_out is None:
            # allocate output array of the same shape (generous)
            sig_out = np.empty_like(sig_in[0], dtype=int32)

        # siglen has one dimension less (the last)
        s = sig_in[0].shape
        siglen = np.empty(s[:-1], dtype=uint32)

        if len(sig_in[0]) == 0:
            return sig_out, siglen

        # call low-level routine
        uleb128_zigzag_diff_array_decode(sig_in[0], sig_in[1], sig_out, siglen)

        return sig_out, siglen

    elif isinstance(sig_in, lgdo.ArrayOfEncodedEqualSizedArrays):
        if not sig_out:
            # initialize output structure with decoded_size
            sig_out = lgdo.ArrayOfEqualSizedArrays(
                dims=(1, 1),
                shape=(len(sig_in), sig_in.decoded_size.value),
                dtype=int32,
                attrs=sig_in.getattrs(),
            )

        siglen = np.empty(len(sig_in), dtype=uint32)
        # save original encoded vector lengths
        nbytes = np.diff(sig_in.encoded_data.cumulative_length.nda, prepend=uint32(0))

        if len(sig_in) == 0:
            return sig_out

        # convert vector of vectors to array of equal sized arrays
        # can now decode on the 2D matrix together with number of bytes to read per row
        _, siglen = decode(
            (sig_in.encoded_data.to_aoesa(preserve_dtype=True).nda, nbytes), sig_out.nda
        )

        # sanity check
        assert np.all(sig_in.decoded_size.value == siglen)

        return sig_out

    elif isinstance(sig_in, lgdo.VectorOfEncodedVectors):
        if sig_out:
            log.warning(
                "a pre-allocated VectorOfVectors was given "
                "to hold an encoded VectorOfVectors. "
                "This is not supported at the moment, so a new one "
                "will be allocated to replace it"
            )

        siglen = np.empty(len(sig_in), dtype=uint32)
        # save original encoded vector lengths
        nbytes = np.diff(sig_in.encoded_data.cumulative_length.nda, prepend=uint32(0))

        # convert vector of vectors to array of equal sized arrays
        # can now decode on the 2D matrix together with number of bytes to read per row
        sig_out, siglen = decode(
            (sig_in.encoded_data.to_aoesa(preserve_dtype=True).nda, nbytes)
        )

        # sanity check
        assert np.array_equal(sig_in.decoded_size, siglen)

        # converto to VOV before returning
        return sig_out.to_vov(np.cumsum(siglen, dtype=uint32))

    else:
        raise ValueError("unsupported input signal type")


@numba.vectorize(
    ["uint64(int64)", "uint32(int32)", "uint16(int16)"],
    nopython=True,
)
def zigzag_encode(x: int | NDArray[int]) -> int | NDArray[int]:
    """ZigZag-encode [#WikiZZ]_ signed integer numbers."""
    return (x >> 31) ^ (x << 1)


@numba.vectorize(
    ["int64(uint64)", "int32(uint32)", "int16(uint16)"],
    nopython=True,
)
def zigzag_decode(x: int | NDArray[int]) -> int | NDArray[int]:
    """ZigZag-decode [#WikiZZ]_ signed integer numbers."""
    return (x >> 1) ^ -(x & 1)


@numba.jit(["uint32(int64, byte[:])"], nopython=True)
def uleb128_encode(x: int, encx: NDArray[ubyte]) -> int:
    """Compute a variable-length representation of an unsigned integer.

    Implements the Unsigned Little Endian Base-128 encoding [#WikiULEB128]_.
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


@numba.jit(["UniTuple(uint32, 2)(byte[:])"], nopython=True)
def uleb128_decode(encx: NDArray[ubyte]) -> (int, int):
    """Decode a variable-length integer into an unsigned integer.

    Implements the Unsigned Little Endian Base-128 decoding [#WikiULEB128]_.
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
    if len(encx) <= 0:
        raise ValueError("input bytes array is empty")

    x = pos = uint32(0)
    for b in encx:
        x = x | ((b & 0x7F) << pos)
        if (b & 0x80) == 0:
            return (x, int(pos / 7 + 1))
        else:
            pos += 7

        if pos >= 64:
            raise OverflowError("overflow during decoding of varint encoded number")

    raise RuntimeError("malformed varint")


@numba.guvectorize(
    [
        "void(uint16[:], byte[:], uint32[:])",
        "void(uint32[:], byte[:], uint32[:])",
        "void(uint64[:], byte[:], uint32[:])",
        "void(int16[:], byte[:], uint32[:])",
        "void(int32[:], byte[:], uint32[:])",
        "void(int64[:], byte[:], uint32[:])",
    ],
    "(n),(m),()",
    nopython=True,
)
def uleb128_zigzag_diff_array_encode(
    sig_in: NDArray[int], sig_out: NDArray[ubyte], nbytes: int
) -> None:
    """Encode an array of integer numbers.

    The algorithm computes the derivative (prepending 0 first) of `sig_in`,
    maps it to positive numbers by applying :func:`zigzag_encode` and finally
    computes its variable-length binary representation with
    :func:`uleb128_encode`.

    The encoded data is stored in `sig_out` as an array of bytes. The number of
    bytes written is stored in `nbytes`. The actual encoded data can therefore
    be found in ``sig_out[:nbytes]``.

    Parameters
    ----------
    sig_in
        the input array of integers.
    sig_out
        pre-allocated bytes array for the output encoded data.
    nbytes
        pre-allocated output array holding the number of bytes written (stored
        in the first index).

    See Also
    --------
    .uleb128_zigzag_diff_array_decode
    """
    pos = uint32(0)
    last = int32(0)
    for s in sig_in:
        zzdiff = zigzag_encode(int32(s - last))
        pos += uleb128_encode(zzdiff, sig_out[pos:])
        last = s

    nbytes[0] = pos


@numba.guvectorize(
    [
        "void(byte[:], uint32[:], uint16[:], uint32[:])",
        "void(byte[:], uint32[:], uint32[:], uint32[:])",
        "void(byte[:], uint32[:], uint64[:], uint32[:])",
        "void(byte[:], uint32[:], int16[:], uint32[:])",
        "void(byte[:], uint32[:], int32[:], uint32[:])",
        "void(byte[:], uint32[:], int64[:], uint32[:])",
    ],
    "(n),(),(m),()",
    nopython=True,
)
def uleb128_zigzag_diff_array_decode(
    sig_in: NDArray[ubyte],
    nbytes: int,
    sig_out: NDArray[int],
    siglen: int,
) -> None:
    """Decode an array of variable-length integers.

    The algorithm inverts :func:`.uleb128_zigzag_diff_array_encode` by decoding
    the variable-length binary data in `sig_in` with :func:`uleb128_decode`,
    then reconstructing the original signal derivative with
    :func:`zigzag_decode` and finally computing its cumulative (i.e. the
    original signal).

    Parameters
    ----------
    sig_in
        the array of bytes encoding the variable-length integers.
    nbytes
        the number of bytes to read from `sig_in` (stored in the first index of
        this array).
    sig_out
        pre-allocated array for the output decoded signal.
    siglen
        the length of the decoded signal, (stored in the first index of this
        array).

    See Also
    --------
    .uleb128_zigzag_diff_array_encode
    """
    if len(sig_in) <= 0:
        raise ValueError("input bytes array is empty")

    _nbytes = min(nbytes[0], len(sig_in))
    pos = i = uint32(0)
    last = int32(0)
    while pos < _nbytes:
        x, nread = uleb128_decode(sig_in[pos:])
        sig_out[i] = last = zigzag_decode(x) + last
        i += 1
        pos += nread

    siglen[0] = i
