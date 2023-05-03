from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
from numpy import int16, int32, ubyte, uint16, uint32
from numpy.typing import NDArray

from pygama import lgdo

from .base import WaveformCodec

# fmt: off
_radware_sigcompress_mask = uint16([0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023,
                                   2047, 4095, 8191, 16383, 32767, 65535])
# fmt: on


@dataclass(frozen=True)
class RadwareSigcompress(WaveformCodec):
    """`radware-sigcompress` array codec.

    Examples
    --------
    >>> from pygama.lgdo.compression import RadwareSigcompress
    >>> codec = RadwareSigcompress(codec_shift=-32768)
    """

    codec_shift: int = 0
    """Offset added to the input waveform before encoding.

    The `radware-sigcompress` algorithm is limited to encoding of 16-bit
    integer values. In certain cases (notably, with *unsigned* 16-bit integer
    values), shifting incompatible data by a fixed amount circumvents the
    issue.
    """


def encode(
    sig_in: NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays,
    sig_out: NDArray[ubyte]
    | lgdo.VectorOfEncodedVectors
    | lgdo.ArrayOfEncodedEqualSizedArrays = None,
    shift: int32 = 0,
) -> NDArray[ubyte] | lgdo.VectorOfEncodedVectors:
    """Compress digital signal(s) with `radware-sigcompress`.

    Wraps :func:`._radware_sigcompress_encode` and adds support for encoding
    LGDO arrays. Resizes the encoded array to its actual length.

    Note
    ----
    The compression algorithm internally interprets the input waveform values as
    16-bit integers. Make sure that your signal can be safely cast to such a
    numeric type. If not, you may want to apply a `shift` to the waveform.

    Parameters
    ----------
    sig_in
        array(s) holding the input signal(s).
    sig_out
        pre-allocated unsigned 8-bit integer array(s) for the compressed
        signal(s). If not provided, a new one will be allocated.
    shift
        value to be added to `sig_in` before compression.

    Returns
    -------
    sig_out
        given pre-allocated `sig_out` structure or new structure of unsigned
        8-bit integers.

    See Also
    --------
    ._radware_sigcompress_encode
    """
    # the encoded signal is an array of bytes -> twice as long as a uint16
    # array
    max_out_len = 2 * len(sig_in)
    if isinstance(sig_in, np.ndarray) and sig_in.ndim == 1:
        if len(sig_in) == 0:
            return sig_in

        if not sig_out:
            # pre-allocate ubyte (uint8) array
            sig_out = np.empty(max_out_len, dtype=ubyte)

        if sig_out.dtype != ubyte:
            raise ValueError("sig_out must be of type ubyte")

        outlen = _radware_sigcompress_encode(sig_in, sig_out, shift=shift)

        # resize (down) the encoded signal to its actual length
        # TODO: really even if user supplied? Maybe not
        if outlen < max_out_len:
            sig_out.resize(outlen, refcheck=True)

    elif isinstance(sig_in, lgdo.ArrayOfEqualSizedArrays):
        if not sig_out:
            # pre-allocate output structure
            # use maximum length possible
            sig_out = lgdo.ArrayOfEncodedEqualSizedArrays(
                encoded_data=lgdo.VectorOfVectors(
                    shape_guess=(len(sig_in), 2 * sig_in.nda.shape[1]), dtype=ubyte
                ),
                decoded_size=sig_in.nda.shape[1],
            )
        elif not isinstance(sig_out, lgdo.ArrayOfEncodedEqualSizedArrays):
            raise ValueError("sig_out must be an ArrayOfEncodedEqualSizedArrays")

        # use unsafe set_vector to fill pre-allocated memory
        for i, wf in enumerate(sig_in):
            sig_out.encoded_data._set_vector_unsafe(i, encode(wf, shift=shift))

        # resize down flattened data array
        # TODO: really even if user supplied? Maybe not
        sig_out.resize(len(sig_in))

    elif isinstance(sig_in, lgdo.VectorOfVectors):
        if not sig_out:
            max_out_len = 2 * len(sig_in.flattened_data) / len(sig_in.cumulative_length)
            sig_out = lgdo.VectorOfEncodedVectors(
                encoded_data=lgdo.VectorOfVectors(
                    shape_guess=(len(sig_in), max_out_len), dtype=ubyte
                ),
            )
        elif not isinstance(sig_out, lgdo.VectorOfEncodedVectors):
            raise ValueError("sig_out must be a VectorOfEncodedVectors")

        # use unsafe set_vector to fill pre-allocated memory
        # should be fast enough
        for i, wf in enumerate(sig_in):
            sig_out.encoded_data._set_vector_unsafe(i, encode(wf, shift=shift))
            sig_out.decoded_size[i] = len(wf)

    else:
        raise ValueError(f"unsupported input signal type ({type(sig_in)})")

        # resize down flattened data array
        # TODO: really even if user supplied? Maybe not
        sig_out.resize(len(sig_in))

    return sig_out


def decode(
    sig_in: NDArray[ubyte]
    | lgdo.VectorOfEncodedVectors
    | lgdo.ArrayOfEncodedEqualSizedArrays,
    sig_out: NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays = None,
    shift: int32 = 0,
) -> NDArray | lgdo.VectorOfVectors | lgdo.ArrayOfEqualSizedArrays:
    """Decompress digital signal(s) with `radware-sigcompress`.

    Wraps :func:`._radware_sigcompress_decode` and adds support for decoding
    LGDOs. Resizes the decoded signals to their actual length.

    Parameters
    ----------
    sig_in
        array(s) holding the input, compressed signal(s).
    sig_out
        pre-allocated array(s) for the decompressed signal(s).  If not
        provided, will allocate a 32-bit integer array(s) structure.
    shift
        the value the original signal(s) was shifted before compression.  The
        value is *subtracted* from samples in `sig_out` right after decoding.

    Returns
    -------
    sig_out
        given pre-allocated structure or new structure of 32-bit integers.

    See Also
    --------
    ._radware_sigcompress_decode
    """
    if isinstance(sig_in, np.ndarray) and sig_in.ndim == 1 and sig_in.dtype == ubyte:
        if len(sig_in) == 0:
            return sig_in

        siglen = _get_hton_u16(sig_in, 0)
        if not sig_out:
            # pre-allocate memory, use safe int32
            sig_out = np.empty(siglen, dtype="int32")
        elif len(sig_out) < siglen:
            # TODO: really even if user supplied? Maybe not
            sig_out.resize(siglen, refcheck=False)

        _radware_sigcompress_decode(sig_in, sig_out, shift=shift)

    elif isinstance(sig_in, lgdo.ArrayOfEncodedEqualSizedArrays):
        if not sig_out:
            # pre-allocate output structure
            sig_out = lgdo.ArrayOfEqualSizedArrays(
                dims=(1, 1),
                shape=(len(sig_in), sig_in.decoded_size.value),
                dtype="int32",
            )

        elif not isinstance(sig_out, lgdo.ArrayOfEqualSizedArrays):
            raise ValueError("sig_out must be an ArrayOfEqualSizedArrays")

        for i, wf in enumerate(sig_in):
            sig_out[i] = decode(wf, shift=shift)

    elif isinstance(sig_in, lgdo.VectorOfEncodedVectors):
        if not sig_out:
            # pre-allocate output structure
            sig_out = lgdo.VectorOfVectors(
                cumulative_length=np.cumsum(sig_in.decoded_size), dtype="int32"
            )

        elif not isinstance(sig_out, lgdo.VectorOfVectors):
            raise ValueError("sig_out must be a VectorOfVectors")

        for i, wf in enumerate(sig_in):
            sig_out[i] = decode(wf[0], shift=shift)

    else:
        raise ValueError(f"unsupported input signal type ({type(sig_in)})")

    return sig_out


@numba.jit(nopython=True)
def _set_hton_u16(a: NDArray[ubyte], i: int, x: int) -> int:
    """Store an unsigned 16-bit integer value in an array of unsigned 8-bit integers.

    The first two most significant bytes from `x` are stored contiguously in
    `a` with big-endian order.
    """
    x_u16 = uint16(x)
    i_1 = i * 2
    i_2 = i_1 + 1
    a[i_1] = ubyte(x_u16 >> 8)
    a[i_2] = ubyte(x_u16 >> 0)
    return x


@numba.jit(nopython=True)
def _get_hton_u16(a: NDArray[ubyte], i: int) -> uint16:
    """Read unsigned 16-bit integer values from an array of unsigned 8-bit integers.

    The first two most significant bytes of the values must be stored
    contiguously in `a` with big-endian order.
    """
    i_1 = i * 2
    i_2 = i_1 + 1
    return uint16(a[i_1] << 8 | a[i_2])


@numba.jit("uint16(uint32)", nopython=True)
def _get_high_u16(x: uint32) -> uint16:
    return uint16(x >> 16)


@numba.jit("uint32(uint32, uint16)", nopython=True)
def _set_high_u16(x: uint32, y: uint16) -> uint32:
    return uint32(x & 0x0000FFFF | (y << 16))


@numba.jit("uint16(uint32)", nopython=True)
def _get_low_u16(x: uint32) -> uint16:
    return uint16(x >> 0)


@numba.jit("uint32(uint32, uint16)", nopython=True)
def _set_low_u16(x: uint32, y: uint16) -> uint32:
    return uint32(x & 0xFFFF0000 | (y << 0))


@numba.jit(nopython=True)
def _radware_sigcompress_encode(
    sig_in: NDArray,
    sig_out: NDArray[ubyte],
    shift: int32,
    _mask: NDArray[uint16] = _radware_sigcompress_mask,
) -> int32:
    """Compress a digital signal.

    Shifts the signal values by ``+shift`` and internally interprets the result
    as :any:`numpy.int16`. Shifted signals must be therefore representable as
    :any:`numpy.int16`, for lossless compression.

    Note
    ----
    The algorithm also computes the first derivative of the input signal, which
    cannot always be represented as a 16-bit integer. In such cases, overflows
    occur, but they seem to be innocuous.

    Almost literal translations of ``compress_signal()`` from the
    `radware-sigcompress` v1.0 C-code by David Radford [1]_. Summary of
    changes:

    - Shift the input signal by `shift` before encoding.
    - Store encoded, :class:`numpy.uint16` signal as an array of bytes
      (:class:`numpy.ubyte`), in big-endian ordering.
    - Declare mask globally to avoid extra memory allocation.
    - Apply just-in-time compilation with Numba.
    - Add a couple of missing array boundary checks.

    .. [1] `radware-sigcompress source code
       <https://legend-exp.github.io/legend-data-format-specs/dev/data_compression/#radware-sigcompress-1>`_.
       released under MIT license `[Copyright (c) 2018, David C. Radford
       <radforddc@ornl.gov>]`.

    Parameters
    ----------
    sig_in
        array of integers holding the input signal. In the original C code,
        an array of 16-bit integers was expected.
    sig_out
        pre-allocated array for the unsigned 8-bit encoded signal. In the
        original C code, an array of unsigned 16-bit integers was expected.

    Returns
    -------
    length
        number of bytes in the encoded signal
    """
    mask = _mask

    i = j = max1 = max2 = min1 = min2 = ds = int16(0)
    nb1 = nb2 = iso = nw = bp = dd1 = dd2 = int16(0)
    dd = uint32(0)

    _set_hton_u16(sig_out, iso, sig_in.size)

    iso += 1
    while j < sig_in.size:  # j = starting index of section of signal
        # find optimal method and length for compression
        # of next section of signal
        si_j = int16(sig_in[j] + shift)
        max1 = min1 = si_j
        max2 = int32(-16000)
        min2 = int32(16000)
        nb1 = nb2 = 2
        nw = 1
        i = j + 1
        # FIXME: 48 could be tuned better?
        while (i < sig_in.size) and (i < j + 48):
            si_i = int16(sig_in[i] + shift)
            si_im1 = int16(sig_in[i - 1] + shift)
            if max1 < si_i:
                max1 = si_i
            if min1 > si_i:
                min1 = si_i
            ds = si_i - si_im1
            if max2 < ds:
                max2 = ds
            if min2 > ds:
                min2 = ds
            nw += 1
            i += 1
        if max1 - min1 <= max2 - min2:  # use absolute values
            nb2 = 99
            while (max1 - min1) > mask[nb1]:
                nb1 += 1
            while (i < sig_in.size) and (
                i < j + 128
            ):  # FIXME: 128 could be tuned better?
                si_i = int16(sig_in[i] + shift)
                if max1 < si_i:
                    max1 = si_i
                dd1 = max1 - min1
                if min1 > si_i:
                    dd1 = max1 - si_i
                if dd1 > mask[nb1]:
                    break
                if min1 > si_i:
                    min1 = si_i
                nw += 1
                i += 1
        else:  # use difference values
            nb1 = 99
            while max2 - min2 > mask[nb2]:
                nb2 += 1
            while (i < sig_in.size) and (
                i < j + 128
            ):  # FIXME: 128 could be tuned better?
                si_i = int16(sig_in[i] + shift)
                si_im1 = int16(sig_in[i - 1] + shift)
                ds = si_i - si_im1
                if max2 < ds:
                    max2 = ds
                dd2 = max2 - min2
                if min2 > ds:
                    dd2 = max2 - ds
                if dd2 > mask[nb2]:
                    break
                if min2 > ds:
                    min2 = ds
                nw += 1
                i += 1

        if bp > 0:
            iso += 1
        # do actual compression
        _set_hton_u16(sig_out, iso, nw)
        iso += 1
        bp = 0
        if nb1 <= nb2:
            # encode absolute values
            _set_hton_u16(sig_out, iso, nb1)
            iso += 1
            _set_hton_u16(sig_out, iso, uint16(min1))
            iso += 1

            i = iso
            while i <= (iso + nw * nb1 / 16):
                _set_hton_u16(sig_out, i, 0)
                i += 1

            i = j
            while i < j + nw:
                dd = int16(sig_in[i] + shift) - min1  # value to encode
                dd = dd << (32 - bp - nb1)
                _set_hton_u16(
                    sig_out, iso, _get_hton_u16(sig_out, iso) | _get_high_u16(dd)
                )
                bp += nb1
                if bp > 15:
                    iso += 1
                    _set_hton_u16(sig_out, iso, _get_low_u16(dd))
                    bp -= 16
                i += 1

        else:
            # encode derivative / difference values
            _set_hton_u16(sig_out, iso, nb2 + 32)  # bits used for encoding, plus flag
            iso += 1
            _set_hton_u16(sig_out, iso, int16(si_j))  # starting signal value
            iso += 1
            _set_hton_u16(sig_out, iso, int16(min2))  # min value used for encoding
            iso += 1

            i = iso
            while i <= iso + nw * nb2 / 16:
                _set_hton_u16(sig_out, i, 0)
                i += 1

            i = j + 1
            while i < j + nw:
                si_i = int16(sig_in[i] + shift)
                si_im1 = int16(sig_in[i - 1] + shift)
                dd = si_i - si_im1 - min2
                dd = dd << (32 - bp - nb2)
                _set_hton_u16(
                    sig_out, iso, _get_hton_u16(sig_out, iso) | _get_high_u16(dd)
                )
                bp += nb2
                if bp > 15:
                    iso += 1
                    _set_hton_u16(sig_out, iso, _get_low_u16(dd))
                    bp -= 16
                i += 1
        j += nw

    if bp > 0:
        iso += 1

    if iso % 2 > 0:
        iso += 1

    return 2 * iso  # number of bytes in compressed signal data


@numba.jit(nopython=True)
def _radware_sigcompress_decode(
    sig_in: NDArray[ubyte],
    sig_out: NDArray,
    shift: int32,
    _mask: NDArray[uint16] = _radware_sigcompress_mask,
) -> int32:
    """Deompress a digital signal.

    After decoding, the signal values are shifted by ``-shift`` to restore the
    original waveform. The dtype of `sig_out` must be large enough to contain it.

    Almost literal translations of ``decompress_signal()`` from the
    `radware-sigcompress` v1.0 C-code by David Radford [1]_. See
    :func:`._radware_sigcompress_encode` for a list of changes to the original
    algorithm.

    Parameters
    ----------
    sig_in
        array holding the input, compressed signal. In the original code, an
        array of 16-bit unsigned integers was expected.
    sig_out
        pre-allocated array for the decompressed signal. In the original code,
        an array of 16-bit integers was expected.

    Returns
    -------
    length
        length of output, decompressed signal.
    """
    mask = _mask

    i = j = min_val = nb = isi = iso = nw = bp = int16(0)
    dd = uint32(0)

    sig_len_in = int(sig_in.size / 2)
    siglen = int16(_get_hton_u16(sig_in, isi))  # signal length
    isi += 1

    while (isi < sig_len_in) and (iso < siglen):
        if bp > 0:
            isi += 1
        bp = 0  # bit pointer
        nw = _get_hton_u16(sig_in, isi)  # number of samples encoded in this chunk
        isi += 1
        nb = _get_hton_u16(sig_in, isi)  # number of bits used in compression
        isi += 1

        if nb < 32:
            # decode absolute values
            min_val = int16(_get_hton_u16(sig_in, isi))  # min value used for encoding
            isi += 1
            dd = _set_low_u16(dd, _get_hton_u16(sig_in, isi))
            i = 0
            while (i < nw) and (iso < siglen):
                if (bp + nb) > 15:
                    bp -= 16
                    dd = _set_high_u16(dd, _get_hton_u16(sig_in, isi))
                    isi += 1
                    if isi < sig_len_in:
                        dd = _set_low_u16(dd, _get_hton_u16(sig_in, isi))
                    dd = dd << (bp + nb)
                else:
                    dd = dd << nb
                sig_out[iso] = (_get_high_u16(dd) & mask[nb]) + min_val - shift
                iso += 1
                bp += nb
                i += 1
        else:
            nb -= 32
            #  decode derivative / difference values
            sig_out[iso] = (
                int16(_get_hton_u16(sig_in, isi)) - shift
            )  # starting signal value
            iso += 1
            isi += 1
            min_val = int16(_get_hton_u16(sig_in, isi))  # min value used for encoding
            isi += 1
            if isi < sig_len_in:
                dd = _set_low_u16(dd, _get_hton_u16(sig_in, isi))

            i = 1
            while (i < nw) and (iso < siglen):
                if (bp + nb) > 15:
                    bp -= 16
                    dd = _set_high_u16(dd, _get_hton_u16(sig_in, isi))
                    isi += 1
                    if isi < sig_len_in:
                        dd = _set_low_u16(dd, _get_hton_u16(sig_in, isi))
                    dd = dd << (bp + nb)
                else:
                    dd = dd << nb
                sig_out[iso] = (
                    int16(
                        (_get_high_u16(dd) & mask[nb])
                        + min_val
                        + sig_out[iso - 1]
                        + shift
                    )
                    - shift
                )
                iso += 1
                bp += nb
                i += 1
        j += nw

    if siglen != iso:
        raise RuntimeError("failure: unexpected signal length after decompression")

    return siglen  # number of shorts in decompressed signal data
