"""Google's data encoding algorithms.

Full specification at <https://protobuf.dev/programming-guides/encoding>.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numba
import numpy as np
from numpy import ubyte, uint32
from numpy.typing import NDArray

from pygama import lgdo

from .base import WaveformCodec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoogleProtobuf(WaveformCodec):
    """Google's Protocol Buffers variable-width integer (*varint*) encoding `[ref] <https://protobuf.dev/programming-guides/encoding>`_."""

    zigzag: bool = True
    """Enable ZigZag encoding."""


def encode(
    sig_in: NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays,
    sig_out: NDArray[ubyte] | lgdo.VectorOfEncodedVectors = None,
    zigzag: bool = False,
) -> (NDArray[ubyte], NDArray[uint32]) | lgdo.VectorOfEncodedVectors:
    """Compress digital signal(s) with Google's Protobuf encoding.

    Wraps :func:`.unsigned_varint_array_encode` and adds support for encoding
    LGDOs.

    Parameters
    ----------
    sig_in
        array(s) holding the input signal(s).
    sig_out
        pre-allocated unsigned 8-bit integer array(s) for the compressed
        signal(s). If not provided, a new one will be allocated.
    zigzag
        whether to apply ZigZag encoding for signed integers.

    Returns
    -------
    sig_out, nbytes
        given pre-allocated `sig_out` structure or new structure of unsigned
        8-bit integers, plus ...

    See Also
    --------
    .unsigned_varint_array_encode
    """
    if len(sig_in) == 0:
        return sig_in

    if isinstance(sig_in, np.ndarray):
        s = sig_in.shape
        if sig_out is None:
            # the encoded signal is an array of bytes
            # at maximum all bytes are used from the input type
            max_b = int(np.iinfo(sig_in.dtype).bits / 8) + 1
            # pre-allocate ubyte (uint8) array
            # expand last dimension
            sig_out = np.empty(s[:-1] + (s[-1] * max_b,), dtype=ubyte)

        if sig_out.dtype != ubyte:
            raise ValueError("sig_out must be of type ubyte")

        # nbytes has one dimension less (the last one)
        nbytes = np.empty(s[:-1], dtype=uint32)

        # might want to zig-zag before making varints
        unsigned_varint_array_encode(
            zigzag_encode(sig_in) if zigzag else sig_in, sig_out, nbytes
        )

        return sig_out, nbytes

    elif isinstance(sig_in, lgdo.VectorOfVectors):
        # convert VectorOfVectors to ArrayOfEqualSizedArrays so it can be
        # directly passed to the low-level encoding routine
        return encode(sig_in.to_aoesa(), sig_out, zigzag=zigzag)

    elif isinstance(sig_in, lgdo.ArrayOfEqualSizedArrays):
        if sig_out:
            log.warning(
                "a pre-allocated VectorOfEncodedVectors was given "
                "to hold an encoded ArrayOfEqualSizedArrays. "
                "This is not supported at the moment, so a new one "
                "will be allocated to replace it"
            )

        # encode the internal numpy array
        sig_out_nda, nbytes = encode(sig_in.nda, zigzag=zigzag)

        # build the encoded LGDO
        encoded_data = lgdo.ArrayOfEqualSizedArrays(nda=sig_out_nda).to_vov(
            cumulative_length=np.cumsum(nbytes, dtype=uint32)
        )
        decoded_size = lgdo.Array(
            np.full(
                shape=(sig_in.nda.shape[0],),
                fill_value=sig_in.nda.shape[1],
                dtype=uint32,
            )
        )

        sig_out = lgdo.VectorOfEncodedVectors(encoded_data, decoded_size)

        return sig_out

    elif isinstance(sig_in, lgdo.Array):
        # encode the internal numpy array
        sig_out_nda, nbytes = encode(sig_in.nda, sig_out, zigzag=zigzag)
        return lgdo.Array(sig_out_nda), nbytes

    else:
        raise ValueError(f"unsupported input signal type ({type(sig_in)})")


def decode(
    sig_in: (NDArray[ubyte], NDArray[uint32]) | lgdo.VectorOfEncodedVectors,
    sig_out: NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays = None,
    zigzag: bool = False,
) -> NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays:
    """Deompress digital signal(s) with Google's Protobuf encoding.

    Wraps :func:`.unsigned_varint_array_decode` and adds support for decoding
    LGDOs. Resizes the decoded signals to their actual length.

    Parameters
    ----------
    sig_in
        array(s) holding the input, compressed signal(s). Output of
        :func:`.encode`.
    sig_out
        pre-allocated array(s) for the decompressed signal(s).  If not
        provided, will allocate a 32-bit integer array(s) structure.
    zigzag
        whether to apply ZigZag decoding after varint decoding.

    Returns
    -------
    sig_out
        given pre-allocated structure or new structure of 32-bit integers.

    See Also
    --------
    ._radware_sigcompress_decode
    """
    if len(sig_in) == 0:
        return sig_in

    # expect the output of encode()
    if isinstance(sig_in, tuple):
        if sig_out is None:
            # allocate output array of the same shape (generous)
            sig_out = np.empty_like(sig_in[0], dtype=uint32)

        # siglen has one dimension less (the last)
        s = sig_in[0].shape
        siglen = np.empty(s[:-1], dtype=uint32)
        # call low-level routine
        unsigned_varint_array_decode(sig_in[0], sig_in[1], sig_out, siglen)

        if zigzag:
            sig_out = zigzag_decode(sig_out)

        return sig_out, siglen

    elif isinstance(sig_in, lgdo.VectorOfEncodedVectors):
        if sig_out:
            log.warning(
                "a pre-allocated LGDO was given "
                "to hold a decoded VectorOfEncodedVectors. "
                "This is not supported at the moment, so a new one "
                "will be allocated to replace it"
            )

        # convert vector of vectors to array of equal sized arrays
        siglen = np.empty(len(sig_in), dtype=uint32)
        # save original encoded vector lengths
        nbytes = np.diff(sig_in.encoded_data.cumulative_length.nda, prepend=uint32(0))

        # can now decode on the 2D matrix together with number of bytes to read per row
        sig_out, siglen = decode(
            (sig_in.encoded_data.to_aoesa(preserve_dtype=True).nda, nbytes),
            zigzag=zigzag,
        )

        # sanity check
        assert np.array_equal(sig_in.decoded_size, siglen)

        # TODO: attributes
        return lgdo.ArrayOfEqualSizedArrays(
            nda=sig_out, attrs=sig_in.getattrs()
        ).to_vov(np.cumsum(siglen, dtype=uint32))

    else:
        raise ValueError("unsupported input signal type")


@numba.vectorize(
    [
        "uint8(int8)",
        "uint16(int16)",
        "uint32(int32)",
        "uint64(int64)",
    ]
)
def zigzag_encode(x: int | NDArray[int]) -> int | NDArray[int]:
    """ZigZag-encode integer numbers."""
    return (x >> 31) ^ (x << 1)


@numba.vectorize(
    [
        "int8(uint8)",
        "int16(uint16)",
        "int32(uint32)",
        "int64(uint64)",
    ]
)
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
    if len(encx) <= 0:
        raise ValueError("input bytes array is empty")

    x = 0
    pos = 0
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
    ],
    "(n),(m),()",
)
def unsigned_varint_array_encode(
    sig_in: NDArray[int], sig_out: NDArray[ubyte], nbytes: int
) -> None:
    """Encode an array of unsigned integer numbers.

    The number of bytes written is stored in `nbytes`. The actual encoded data
    is found in ``sig_out[:nbytes]``.

    Parameters
    ----------
    sig_in
        the input array of integers.
    sig_out
        pre-allocated array for the output encoded data.
    nbytes
        pre-allocated output array holding the number of bytes written.

    See Also
    --------
    .unsigned_varint_encode, .unsigned_varint_array_decode
    """
    pos = 0
    for s in sig_in:
        pos += unsigned_varint_encode(s, sig_out[pos:])
    nbytes[0] = pos


@numba.guvectorize(
    [
        "void(byte[:], uint32[:], uint16[:], uint32[:])",
        "void(byte[:], uint32[:], uint32[:], uint32[:])",
        "void(byte[:], uint32[:], uint64[:], uint32[:])",
    ],
    "(n),(),(m),()",
)
def unsigned_varint_array_decode(
    sig_in: NDArray[ubyte],
    nbytes: int,
    sig_out: NDArray[int],
    siglen: int,
) -> None:
    """Decode an array of varints, as returned by :func:`.unsigned_varint_array_encode`.

    Parameters
    ----------
    sig_in
        the array of varints.
    nbytes
        the number of bytes to read from `sig_in`, stored in the first index of
        this array.
    sig_out
        pre-allocated array for the output decoded integers.
    siglen
        the length of the decoded array, stored in the first index of this
        array.

    See Also
    --------
    .unsigned_varint_decode, .unsigned_varint_array_encode
    """
    if len(sig_in) <= 0:
        raise ValueError("input bytes array is empty")

    _nbytes = min(nbytes[0], len(sig_in))
    pos = i = 0
    while pos < _nbytes:
        x, nread = unsigned_varint_decode(sig_in[pos:])
        pos += nread
        sig_out[i] = x
        i += 1
    siglen[0] = i
