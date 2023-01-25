"""Data compression utilities."""

from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray

from pygama.lgdo import WaveformTable

# fmt: off
_radware_siggen_mask = np.uint16([0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023,
                                  2047, 4095, 8191, 16383, 32767, 65535])
# fmt: on


def radware_compress(
    sig_in: NDArray | WaveformTable, sig_out: NDArray | WaveformTable = None
) -> NDArray | WaveformTable:
    """Compress digital signal(s) with `radware-sigcompress`.

    Parameters
    ----------
    sig_in
        array or waveform table holding the input signal(s).
    sig_out
        pre-allocated array or waveform table for the compressed signal(s).

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
            sig_out = np.empty_like(sig_in, dtype=np.uint16)

        max_out_len = 2 * sig_in.size
        if len(sig_out) < max_out_len:
            sig_out.resize(max_out_len, refcheck=False)

        outlen = _radware_sigcompress_encode(sig_in, sig_out)

        if outlen < sig_in.size:
            sig_out.resize(outlen, refcheck=False)

    elif isinstance(sig_in, WaveformTable):
        if not sig_out:
            # pre-allocate output table
            # sig_out.values will be a VectorOfVectors
            sig_out = WaveformTable(
                size=sig_in.size,
                dtype=np.uint16,
                t0=sig_in.t0,
                dt=sig_in.dt,
                attrs=sig_in.attrs,
            )

        for i, wf in enumerate(sig_in.values):
            sig_out.values.set_vector(i, radware_compress(wf))

    else:
        raise ValueError(f"unsupported input signal type ({type(sig_in)})")

    return sig_out


def radware_decompress(
    sig_in: NDArray | WaveformTable, sig_out: NDArray | WaveformTable = None
) -> NDArray | WaveformTable:
    """Decompress digital signal(s) with `radware-sigcompress`.

    Parameters
    ----------
    sig_in
        array or waveform table holding the input, compressed signal(s).
    sig_out
        pre-allocated array or waveform table for the decompressed signal(s).

    Returns
    -------
    sig_out
        given pre-allocated structure or structure of 16-bit integers.

    See Also
    --------
    ._radware_decompress_decode
    """
    if isinstance(sig_in, np.ndarray) and sig_in.ndim == 1:
        siglen = int(sig_in[0])
        if not sig_out:
            # pre-allocate memory
            sig_out = np.empty(siglen, dtype=np.int16)
        elif len(sig_out) < siglen:
            sig_out.resize(siglen, refcheck=False)

        outlen = _radware_sigcompress_decode(sig_in, sig_out)

        if outlen < len(sig_out):
            sig_out.resize(outlen, refcheck=False)

    elif isinstance(sig_in, WaveformTable):
        if not sig_out:
            # pre-allocate  output table
            # sig_out.values will be a VectorOfVectors because that's the most
            # general format
            sig_out = WaveformTable(
                size=len(sig_in),
                dtype=np.int16,
                t0=sig_in.t0,
                dt=sig_in.dt,
                attrs=sig_in.attrs,
            )

        for i, wf in enumerate(sig_in.values):
            sig_out.values.set_vector(i, radware_decompress(wf))

    else:
        raise ValueError(f"unsupported input signal type ({type(sig_in)})")

    return sig_out


@numba.jit(nopython=True)
def _radware_sigcompress_encode(
    sig_in: NDArray,
    sig_out: NDArray,
    _mask: NDArray[np.uint16] = _radware_siggen_mask,
) -> np.int32:
    """Compress a digital signal.

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
    db = np.zeros(2, dtype=np.uint16)
    dd = np.frombuffer(db, dtype=np.uintc)

    iso += 1
    while j < sig_in.size:  # j = starting index of section of signal
        # find optimal method and length for compression
        # of next section of signal
        max1 = min1 = sig_in[j]
        max2 = -16000
        min2 = 16000
        nb1 = nb2 = 2
        nw = 1
        i = j + 1
        # FIXME: 48 could be tuned better?
        while (i < sig_in.size) and (i < j + 48):
            if max1 < sig_in[i]:
                max1 = sig_in[i]
            if min1 > sig_in[i]:
                min1 = sig_in[i]
            ds = sig_in[i] - sig_in[i - 1]
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
                if max1 < sig_in[i]:
                    max1 = sig_in[i]
                dd1 = max1 - min1
                if min1 > sig_in[i]:
                    dd1 = max1 - sig_in[i]
                if dd1 > mask[nb1]:
                    break
                if min1 > sig_in[i]:
                    min1 = sig_in[i]
                nw += 1
                i += 1
        else:  # use difference values
            nb1 = 99
            while max2 - min2 > mask[nb2]:
                nb2 += 1
            while (i < sig_in.size) and (
                i < j + 128
            ):  # FIXME: 128 could be tuned better?
                ds = sig_in[i] - sig_in[i - 1]
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
                dd[0] = sig_in[i] - min1  # value to encode
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
            sig_out[iso] = sig_in[j]  # starting signal value
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


@numba.jit(nopython=True)
def _radware_sigcompress_decode(
    sig_in: NDArray,
    sig_out: NDArray,
    _mask: NDArray[np.uint16] = _radware_siggen_mask,
) -> np.int32:
    """Deompress a digital signal.

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
    siglen = np.uint16(sig_in[isi])  # signal length
    isi += 1
    db = np.zeros(2, dtype=np.uint16)
    dd = np.frombuffer(db, dtype=np.uintc)

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
                    db[0] = sig_in[isi]
                    dd[0] = dd[0] << (bp + nb)
                else:
                    dd[0] = dd[0] << nb
                sig_out[iso] = (db[1] & mask[nb]) + min_val
                iso += 1
                bp += nb
                i += 1
        else:
            nb -= 32
            #  decode derivative / difference values
            sig_out[iso] = np.int16(sig_in[isi])  # starting signal value
            iso += 1
            isi += 1
            min_val = np.int16(sig_in[isi])  # min value used for encoding
            isi += 1
            db[0] = sig_in[isi]

            i = 1
            while (i < nw) and (iso < siglen):
                if (bp + nb) > 15:
                    bp -= 16
                    db[1] = sig_in[isi]
                    isi += 1
                    db[0] = sig_in[isi]
                    dd[0] = dd[0] << (bp + nb)
                else:
                    dd[0] = dd[0] << nb
                sig_out[iso] = (db[1] & mask[nb]) + min_val + sig_out[iso - 1]
                iso += 1
                bp += nb
                i += 1
        j += nw

    if siglen != iso:
        raise RuntimeError("failure: unexpected signal length after decompression")

    return siglen  # number of shorts in decompressed signal data
