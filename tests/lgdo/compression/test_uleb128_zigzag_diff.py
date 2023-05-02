import numpy as np

from pygama.lgdo import ArrayOfEncodedEqualSizedArrays, ArrayOfEqualSizedArrays
from pygama.lgdo.compression import varlen


def _to_bin(array):
    return [np.binary_repr(x, width=8) for x in array]


def test_zigzag():
    assert varlen.zigzag_encode(0) == 0
    assert varlen.zigzag_encode(-1) == 1
    assert varlen.zigzag_encode(1) == 2
    assert varlen.zigzag_encode(-2) == 3
    assert varlen.zigzag_encode(2) == 4

    assert varlen.zigzag_decode(np.uint8(0)) == 0
    assert varlen.zigzag_decode(np.uint8(1)) == -1
    assert varlen.zigzag_decode(np.uint8(2)) == 1
    assert varlen.zigzag_decode(np.uint8(3)) == -2
    assert varlen.zigzag_decode(np.uint8(4)) == 2


def test_varint_encoding():
    # import varint varint
    # >>> ["{0:08b}".format(b) for b in varint.encode(x)]
    expected = {
        1: ["00000001"],
        5: ["00000101"],
        127: ["01111111"],
        128: ["10000000", "00000001"],
        150: ["10010110", "00000001"],
        7896: ["11011000", "00111101"],
        8192: ["10000000", "01000000"],
        268435455: ["11111111", "11111111", "11111111", "01111111"],
    }

    for x, varint in expected.items():
        encx = np.empty(10, dtype="ubyte")
        pos = varlen.uleb128_encode(x, encx)
        assert _to_bin(encx[:pos]) == varint
        assert varlen.uleb128_decode(encx[:pos]) == (x, len(varint))


def test_varint_encode_decode_equality():
    for x in [1, 3856, 234, 11, 93645, 2, 45]:
        encx = np.empty(100, dtype="ubyte")
        pos = varlen.uleb128_encode(x, encx)
        assert varlen.uleb128_decode(encx[:pos])[0] == x


def test_uleb128zzdiff_encode_decode_equality():
    sig_in = np.array([1, 3856, 234, 11, 93645, 2, 45], dtype="uint32")
    sig_out = np.empty(100, dtype="ubyte")
    nbytes = np.empty(1, dtype="uint32")

    varlen.uleb128_zigzag_diff_array_encode(sig_in, sig_out, nbytes)

    encx = np.empty(10, dtype="ubyte")
    offset = 0
    last = 0
    for s in sig_in:
        pos = varlen.uleb128_encode(varlen.zigzag_encode(int(s) - last), encx)
        assert np.array_equal(sig_out[offset : offset + pos], encx[:pos])
        offset += pos
        last = s

    sig_in_dec = np.empty(100, dtype="uint32")
    siglen = np.empty(1, dtype="uint32")
    varlen.uleb128_zigzag_diff_array_decode(sig_out, nbytes, sig_in_dec, siglen)
    assert np.array_equal(sig_in, sig_in_dec[0 : siglen[0]])


def test_uleb128zzdiff_encode_decode_2dmatrix_equality():
    sig_in = np.random.randint(0, 10000, (10, 1000), dtype="uint32")
    sig_out = np.empty((10, 1000 * 4), dtype="ubyte")
    nbytes = np.empty(10, dtype="uint32")

    varlen.uleb128_zigzag_diff_array_encode(sig_in, sig_out, nbytes)

    sig_in_dec = np.empty((10, 1000), dtype="uint32")
    siglen = np.empty(10, dtype="uint32")
    varlen.uleb128_zigzag_diff_array_decode(sig_out, nbytes, sig_in_dec, siglen)

    assert np.array_equal(sig_in, sig_in_dec)


def test_uleb128zzdiff_encode_decode_wrapper_2dmatrix_equality():
    sig_in = np.random.randint(0, 10000, (10, 1000), dtype="uint32")
    sig_in_dec, siglen = varlen.decode(varlen.encode(sig_in))
    assert np.array_equal(siglen, np.full(10, 1000))

    assert np.array_equal(sig_in, sig_in_dec[:, : siglen[0]])

    sig_in_dec = np.empty((10, 1000), dtype="uint32")
    varlen.decode(varlen.encode(sig_in), sig_in_dec)

    assert np.array_equal(sig_in, sig_in_dec)


def test_uleb128zzdiff_encode_decode_lgdo_aoesa(wftable):
    voev = varlen.encode(wftable.values)
    assert isinstance(voev, ArrayOfEncodedEqualSizedArrays)
    assert voev.decoded_size.value == wftable.wf_len

    for i in range(len(voev)):
        wf_enc, nbytes = varlen.encode(wftable.values[i])
        assert len(voev[i]) == nbytes
        assert np.array_equal(voev[i], wf_enc[:nbytes])
        wf_dec, siglen = varlen.decode((wf_enc, nbytes))
        assert np.array_equal(wf_dec[:siglen], wftable.values[i])

    assert voev.encoded_data.to_aoesa(preserve_dtype=True).nda.dtype == np.ubyte

    sig_in_dec = varlen.decode(voev)
    assert isinstance(sig_in_dec, ArrayOfEqualSizedArrays)
    assert sig_in_dec == wftable.values

    # decode with pre-allocated LGDO
    sig_in_dec = ArrayOfEqualSizedArrays(
        dims=(1, 1), shape=(len(wftable), wftable.wf_len), dtype="uint16"
    )
    varlen.decode(voev, sig_in_dec)
    assert sig_in_dec == wftable.values


def test_encoding_decoding_empty():
    aoesa = ArrayOfEqualSizedArrays(dims=(1, 1), nda=np.empty((0, 0)))
    enc_aoesa = varlen.encode(aoesa)
    assert enc_aoesa == ArrayOfEncodedEqualSizedArrays()
    assert varlen.decode(enc_aoesa) == aoesa
