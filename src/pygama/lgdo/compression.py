"""Data compression utilities."""

from __future__ import annotations

import logging

import numba
import numpy as np
from numpy import int16, int32, ubyte, uint16, uintc
from numpy.typing import NDArray

from . import ArrayOfEqualSizedArrays, VectorOfEncodedVectors, VectorOfVectors
from . import lgdo_utils as utils

# fmt: off
_radware_siggen_mask = uint16([0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023,
                               2047, 4095, 8191, 16383, 32767, 65535])
# fmt: on

log = logging.getLogger(__name__)


def radware_compress(
    sig_in: NDArray | VectorOfVectors | ArrayOfEqualSizedArrays,
    sig_out: NDArray | VectorOfEncodedVectors = None,
    shift: int32 = -32768,
) -> NDArray | VectorOfEncodedVectors:
    """Compress digital signal(s) with `radware-sigcompress`.

    Parameters
    ----------
    sig_in
        array or array of arrays holding the input signal(s).
    sig_out
        pre-allocated array or array of arrays for the compressed signal(s).
    shift
        value to be added to `sig_in` before compression.

    Returns
    -------
    sig_out
        given pre-allocated structure or structure of unsigned 16-bit integers.

    See Also
    --------
    ._radware_decompress_encode
    """
    if isinstance(sig_in, np.ndarray) and sig_in.ndim == 1:
        if not sig_out:
            # pre-allocate memory
            sig_out = np.empty_like(sig_in, dtype=uint16)

        if sig_out.dtype != uint16:
            raise ValueError("sig_out must be of type uint16")

        max_out_len = 2 * sig_in.size
        if len(sig_out) < max_out_len:
            sig_out.resize(max_out_len, refcheck=True)

        outlen = _radware_sigcompress_encode(sig_in, sig_out, shift=shift)

        if outlen < sig_in.size:
            sig_out.resize(outlen, refcheck=True)

    # TODO: different actions if ArrayOfEqualSizedArrays
    elif isinstance(sig_in, (VectorOfVectors, ArrayOfEqualSizedArrays)):
        if not sig_out:
            # pre-allocate output structure
            sig_out = VectorOfEncodedVectors(
                encoded_data=VectorOfVectors(
                    shape_guess=(len(sig_in), 1), dtype=uint16
                ),
                attrs={"codec": "radware_sigcompress", "codec_shift": shift},
            )
        elif not isinstance(sig_out, VectorOfEncodedVectors):
            raise ValueError("sig_out must be a VectorOfEncodedVectors")

        for i, wf in enumerate(sig_in):
            sig_out[i] = (radware_compress(wf, shift=shift), len(wf))

    else:
        raise ValueError(f"unsupported input signal type ({type(sig_in)})")

    return sig_out


def radware_decompress(
    sig_in: NDArray | VectorOfEncodedVectors,
    sig_out: NDArray | VectorOfVectors | ArrayOfEqualSizedArrays = None,
    shift: int32 = -32768,
    warn: bool = True,
) -> NDArray | VectorOfVectors | ArrayOfEqualSizedArrays:
    """Decompress digital signal(s) with `radware-sigcompress`.

    Parameters
    ----------
    sig_in
        array or waveform table holding the input, compressed signal(s).
    sig_out
        pre-allocated array or waveform table for the decompressed signal(s).
    shift
        the value the original waveform was shifted before compression.  The
        value is subtracted from samples in `sig_out` right after decoding.

    Returns
    -------
    sig_out
        given pre-allocated structure or structure of integers.

    See Also
    --------
    ._radware_decompress_decode
    """
    if isinstance(sig_in, np.ndarray) and sig_in.ndim == 1 and sig_in.dtype == uint16:
        siglen = int(sig_in[0])
        if not sig_out:
            # pre-allocate memory, use safe int32
            sig_out = np.empty(siglen, dtype=int32)
        elif len(sig_out) < siglen:
            sig_out.resize(siglen, refcheck=False)

        outlen = _radware_sigcompress_decode(sig_in, sig_out, shift=shift)

        if outlen < len(sig_out):
            sig_out.resize(outlen, refcheck=False)

    elif isinstance(sig_in, VectorOfEncodedVectors):
        if not sig_out:
            # pre-allocate output structure
            # sig_out will be a VectorOfVectors for now because that's the most
            # general format
            sig_out = utils.copy(sig_in.encoded_data, dtype=int32)

        elif not isinstance(sig_out, (VectorOfVectors, ArrayOfEqualSizedArrays)):
            raise ValueError(
                "sig_out must be a ArrayOfEqualSizedArrays or VectorOfVectors"
            )

        if "codec_shift" in sig_in.attrs and isinstance(
            sig_in.attrs["codec_shift"], int
        ):
            if sig_in.attrs["codec_shift"] != shift and warn:
                log.warning(
                    f"shift = {shift} != attrs['codec_shift'] = {sig_in.attrs['codec_shift']}. "
                    "The decoded waveform will not correspond to the one originally encoded. "
                    "If you know what you are doing. Suppress this warning by setting warn=False"
                )
            elif shift is None:
                shift = sig_in.attrs["codec_shift"]

        for i, wf in enumerate(sig_in):
            sig_out[i] = radware_decompress(wf[0], shift=shift)

    else:
        raise ValueError(f"unsupported input signal type ({type(sig_in)})")

    return sig_out


@numba.jit()
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


@numba.jit()
def _get_hton_u16(a: NDArray[ubyte], i: int) -> uint16:
    """Read unsigned 16-bit integer values from an array of unsigned 8-bit integers.

    The first two most significant bytes of the values must be stored
    contiguously in `a` with big-endian order.
    """
    i_1 = i * 2
    i_2 = i_1 + 1
    return uint16(a[i_1] << 8 | a[i_2])


@numba.jit()
def _radware_sigcompress_encode(
    sig_in: NDArray[uint16],
    sig_out: NDArray,
    shift: int32 = -32768,
    _mask: NDArray[uint16] = _radware_siggen_mask,
) -> int32:
    """Compress a digital signal.

    Shifts the signal values by ``+shift`` and internally interprets the result
    as :any:`numpy.int16`. Shifted signals must be representable as
    :any:`numpy.int16`.

    Almost literal translations of ``compress_signal()`` from the
    `radware-sigcompress` v1.0 C-code by David Radford [1]_.

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
        pre-allocated array for the compressed signal. In the original C code,
        an array of unsigned 16-bit integers was expected.

    Returns
    -------
    length
        length of output signal.
    """
    mask = _mask

    j = iso = bp = 0
    sig_out[iso] = sig_in.size
    db = np.zeros(2, dtype=uint16)
    dd = np.frombuffer(db, dtype=uintc)

    iso += 1
    while j < sig_in.size:  # j = starting index of section of signal
        # find optimal method and length for compression
        # of next section of signal
        max1 = min1 = int16(sig_in[j] + shift)
        max2 = int16(-16000)
        min2 = int16(16000)
        nb1 = nb2 = 2
        nw = 1
        i = j + 1
        # FIXME: 48 could be tuned better?
        while (i < sig_in.size) and (i < j + 48):
            sig_in_i = int16(sig_in[i] + shift)
            if max1 < sig_in_i:
                max1 = sig_in_i
            if min1 > sig_in_i:
                min1 = sig_in_i
            ds = int16(sig_in[i] - sig_in[i - 1])
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
                sig_in_i = int16(sig_in[i] + shift)
                if max1 < sig_in_i:
                    max1 = sig_in_i
                dd1 = max1 - min1
                if min1 > sig_in_i:
                    dd1 = max1 - sig_in_i
                if dd1 > mask[nb1]:
                    break
                if min1 > sig_in_i:
                    min1 = sig_in_i
                nw += 1
                i += 1
        else:  # use difference values
            nb1 = 99
            while max2 - min2 > mask[nb2]:
                nb2 += 1
            while (i < sig_in.size) and (
                i < j + 128
            ):  # FIXME: 128 could be tuned better?
                ds = int16(sig_in[i] - sig_in[i - 1])
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
        sig_out[iso] = nw
        iso += 1
        bp = 0
        if nb1 <= nb2:
            # encode absolute values
            sig_out[iso] = nb1  # number of bits used for encoding
            iso += 1
            sig_out[iso] = min1  # min value used for encoding
            iso += 1

            i = iso
            while i <= (iso + nw * nb1 / 16):
                sig_out[i] = 0
                i += 1

            i = j
            while i < j + nw:
                dd[0] = int16(sig_in[i] + shift) - min1  # value to encode
                dd[0] = dd[0] << (32 - bp - nb1)
                sig_out[iso] |= db[1]
                bp += nb1
                if bp > 15:
                    iso += 1
                    sig_out[iso] = db[0]
                    bp -= 16
                i += 1

        else:
            # encode derivative / difference values
            sig_out[iso] = nb2 + 32  # bits used for encoding, plus flag
            iso += 1
            sig_out[iso] = sig_in[j] + shift  # starting signal value
            iso += 1
            sig_out[iso] = min2  # min value used for encoding
            iso += 1

            i = iso
            while i <= iso + nw * nb2 / 16:
                sig_out[i] = 0
                i += 1

            i = j + 1
            while i < j + nw:
                dd[0] = sig_in[i] - sig_in[i - 1] - min2  # value to encode
                dd[0] = dd[0] << (32 - bp - nb2)
                sig_out[iso] |= db[1]
                bp += nb2
                if bp > 15:
                    iso += 1
                    sig_out[iso] = db[0]
                    bp -= 16
                i += 1
        j += nw

    if bp > 0:
        iso += 1

    if iso % 2 > 0:
        iso += 1

    return iso  # number of shorts in decompressed signal data


@numba.jit()
def _radware_sigcompress_decode(
    sig_in: NDArray[uint16],
    sig_out: NDArray,
    shift: int32 = -32768,
    _mask: NDArray[uint16] = _radware_siggen_mask,
) -> int32:
    """Deompress a digital signal.

    After decoding, the signal values are shifted by ``-shift`` to restore the
    original waveform. The dtype of `sig_out` must be large enough.

    Almost literal translations of ``decompress_signal()`` from the
    `radware-sigcompress` v1.0 C-code by David Radford [1]_.

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
        length of output signal.
    """
    mask = _mask

    sig_len_in = sig_in.size
    j = isi = iso = bp = 0
    siglen = uint16(sig_in[isi])  # signal length
    isi += 1
    db = np.zeros(2, dtype=uint16)
    dd = np.frombuffer(db, dtype=uintc)

    while (isi < sig_len_in) and (iso < siglen):
        if bp > 0:
            isi += 1
        bp = 0  # bit pointer
        nw = sig_in[isi]  # number of samples encoded in this chunk
        isi += 1
        nb = sig_in[isi]  # number of bits used in compression
        isi += 1

        if nb < 32:
            min_val = sig_in[isi]  # min value used for encoding
            isi += 1
            db[0] = sig_in[isi]
            i = 0
            while (i < nw) and (iso < siglen):
                if (bp + nb) > 15:
                    bp -= 16
                    db[1] = sig_in[isi]
                    isi += 1
                    if isi < sig_len_in:
                        db[0] = sig_in[isi]
                    dd[0] = dd[0] << (bp + nb)
                else:
                    dd[0] = dd[0] << nb
                sig_out[iso] = int16((db[1] & mask[nb]) + min_val) - shift
                iso += 1
                bp += nb
                i += 1
        else:
            nb -= 32
            #  decode derivative / difference values
            sig_out[iso] = int16(sig_in[isi]) - shift  # starting signal value
            iso += 1
            isi += 1
            min_val = int16(sig_in[isi])  # min value used for encoding
            isi += 1
            db[0] = sig_in[isi]

            i = 1
            while (i < nw) and (iso < siglen):
                if (bp + nb) > 15:
                    bp -= 16
                    db[1] = sig_in[isi]
                    isi += 1
                    if isi < sig_len_in:
                        db[0] = sig_in[isi]
                    dd[0] = dd[0] << (bp + nb)
                else:
                    dd[0] = dd[0] << nb
                sig_out[iso] = (
                    int16((db[1] & mask[nb]) + min_val + sig_out[iso - 1] + shift)
                    - shift
                )
                iso += 1
                bp += nb
                i += 1
        j += nw

    if siglen != iso:
        raise RuntimeError("failure: unexpected signal length after decompression")

    return siglen  # number of shorts in decompressed signal data
