"""Data compression utilities."""

from __future__ import annotations

import numba
import numpy as np
from numpy.typing import NDArray


@numba.jit(nopython=True)
def _radware_sigcompress_encode(sig_in: NDArray, sig_out: NDArray) -> int:
    """Compress a digital signal.

    Almost literal translations of ``compress_signal()`` from the
    `radware-sigcompress` v1.0 C-code by David Radford [1]_.

    .. [1] `radware-sigcompress source code
       <https://legend-exp.github.io/legend-data-format-specs/dev/data_compression/#radware-sigcompress-1>`_.
       released under MIT license `[Copyright (c) 2018, David C. Radford <radforddc@ornl.gov>]`.

    Parameters
    ----------
    sig_in
        array holding the input signal.
    sig_out
        pre-allocated array for the compressed signal.

    Returns
    -------
    length
        length of output signal.
    """
    mask = [
        0,
        1,
        3,
        7,
        15,
        31,
        63,
        127,
        255,
        511,
        1023,
        2047,
        4095,
        8191,
        16383,
        32767,
        65535,
    ]

    j = iso = bp = 0
    sig_out[iso] = len(sig_in)
    db = np.zeros(2, dtype=np.ushort)
    dd = np.frombuffer(db, dtype=np.uintc)

    iso += 1
    while j < len(sig_in):  # j = starting index of section of signal
        # find optimal method and length for compression
        # of next section of signal
        max1 = min1 = sig_in[j]
        max2 = -16000
        min2 = 16000
        nb1 = nb2 = 2
        nw = 1
        i = j + 1
        while (i < len(sig_in)) and (i < j + 48):  # FIXME: 48 could be tuned better?
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
            while (i < len(sig_in)) and (
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
            while (i < len(sig_in)) and (
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

    return iso  # number of shorts in decompressed signal data


@numba.jit(nopython=True)
def _radware_sigcompress_decode(sig_in: NDArray, sig_out: NDArray) -> int:
    """Deompress a digital signal.

    Almost literal translations of ``decompress_signal()`` from the
    `radware-sigcompress` v1.0 C-code by David Radford [1]_.

    Parameters
    ----------
    sig_in
        array holding the input, compressed signal.
    sig_out
        pre-allocated array for the decompressed signal.

    Returns
    -------
    length
        length of output signal.
    """
    mask = [
        0,
        1,
        3,
        7,
        15,
        31,
        63,
        127,
        255,
        511,
        1023,
        2047,
        4095,
        8191,
        16383,
        32767,
        65535,
    ]

    sig_len_in = len(sig_in)
    j = isi = iso = bp = 0
    siglen = np.ushort(sig_in[isi])  # signal length
    isi += 1
    db = np.zeros(2, dtype=np.ushort)
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
            sig_out[iso] = np.short(sig_in[isi])  # starting signal value
            iso += 1
            isi += 1
            min_val = np.short(sig_in[isi])  # min value used for encoding
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
