import numpy as np

from pygama.lgdo import VectorOfEncodedVectors
from pygama.lgdo.compression import google


def _to_bin(array):
    return [np.binary_repr(x, width=8) for x in array]


def test_zigzag():
    assert google.zigzag_encode(0) == 0
    assert google.zigzag_encode(-1) == 1
    assert google.zigzag_encode(1) == 2
    assert google.zigzag_encode(-2) == 3
    assert google.zigzag_encode(2) == 4

    assert google.zigzag_decode(np.uint8(0)) == 0
    assert google.zigzag_decode(np.uint8(1)) == -1
    assert google.zigzag_decode(np.uint8(2)) == 1
    assert google.zigzag_decode(np.uint8(3)) == -2
    assert google.zigzag_decode(np.uint8(4)) == 2


def test_varint_encoding():
    encx = np.empty(10, dtype="ubyte")
    pos = google.unsigned_varint_encode(1, encx)
    assert _to_bin(encx[:pos]) == ["00000001"]
    assert google.unsigned_varint_decode(encx[:pos]) == (1, 1)

    pos = google.unsigned_varint_encode(150, encx)
    assert _to_bin(encx[:pos]) == ["10010110", "00000001"]
    assert google.unsigned_varint_decode(encx[:pos]) == (150, 2)


def test_varint_encode_decode_equality():
    for x in [1, 3856, 234, 11, 93645, 2, 45]:
        encx = np.empty(100, dtype="ubyte")
        pos = google.unsigned_varint_encode(x, encx)
        assert google.unsigned_varint_decode(encx[:pos])[0] == x


def test_varint_array_encode_decode_equality():
    sig_in = np.array([1, 3856, 234, 11, 93645, 2, 45], dtype="uint32")
    sig_out = np.empty(100, dtype="ubyte")
    nbytes = np.empty(1, dtype="uint32")

    google.unsigned_varint_array_encode(sig_in, sig_out, nbytes)

    sig_in_dec = np.empty(100, dtype="uint32")
    siglen = np.empty(1, dtype="uint32")
    google.unsigned_varint_array_decode(sig_out, nbytes, sig_in_dec, siglen)
    assert np.array_equal(sig_in, sig_in_dec[0 : siglen[0]])


def test_varint_array_encode_decode_2dmatrix_equality():
    sig_in = np.random.randint(0, 10000, (10, 1000), dtype="uint32")
    sig_out = np.empty((10, 1000 * 4), dtype="ubyte")
    nbytes = np.empty(10, dtype="uint32")

    google.unsigned_varint_array_encode(sig_in, sig_out, nbytes)

    sig_in_dec = np.empty((10, 1000), dtype="uint32")
    siglen = np.empty(10, dtype="uint32")
    google.unsigned_varint_array_decode(sig_out, nbytes, sig_in_dec, siglen)

    assert np.array_equal(sig_in, sig_in_dec)


def test_zigzag_varint_array_encode_decode_equality():
    sig_in = np.array([1, -3856, -234, 11, 93645, -2, 45])
    sig_out = np.empty(100, dtype="ubyte")
    nbytes = np.empty(1, dtype="uint32")

    google.unsigned_varint_array_encode(
        google.zigzag_encode(sig_in).astype("uint32"), sig_out, nbytes
    )

    sig_in_dec = np.empty(100, dtype="uint32")
    siglen = np.empty(1, dtype="uint32")
    google.unsigned_varint_array_decode(sig_out, nbytes, sig_in_dec, siglen)
    assert np.array_equal(sig_in, google.zigzag_decode(sig_in_dec[0 : siglen[0]]))


def test_varint_array_encode_decode_wrapper_2dmatrix_equality():
    sig_in = np.random.randint(0, 10000, (10, 1000), dtype="uint32")
    sig_in_dec, siglen = google.decode(google.encode(sig_in))
    assert np.array_equal(siglen, np.full(10, 1000))

    assert np.array_equal(sig_in, sig_in_dec[:, : siglen[0]])

    sig_in_dec = np.empty((10, 1000), dtype="uint32")
    google.decode(google.encode(sig_in), sig_in_dec)

    assert np.array_equal(sig_in, sig_in_dec)


def test_encode_decode_lgdo_aoesa(wftable):
    voev = google.encode(wftable.values)
    assert isinstance(voev, VectorOfEncodedVectors)
    assert np.array_equal(voev.decoded_size, np.full(len(wftable), wftable.wf_len))

    for i in range(len(voev)):
        wf_enc, nbytes = google.encode(wftable.values[i])
        assert len(voev[i][0]) == nbytes
        assert np.array_equal(voev[i][0], wf_enc[:nbytes])

    assert voev.encoded_data.to_aoesa(preserve_dtype=True).nda.dtype == np.ubyte

    sig_in_dec = google.decode(voev)
    np.array_equal(sig_in_dec, wftable.values)


def test_encode_decode_lgdo_aoesa_zigzag(wftable):
    voev = google.encode(wftable.values, zigzag=True)
    assert isinstance(voev, VectorOfEncodedVectors)
    assert np.array_equal(voev.decoded_size, np.full(len(wftable), wftable.wf_len))

    for i in range(len(voev)):
        wf_enc, nbytes = google.encode(wftable.values[i], zigzag=True)
        assert len(voev[i][0]) == nbytes
        assert np.array_equal(voev[i][0], wf_enc[:nbytes])

    assert voev.encoded_data.to_aoesa(preserve_dtype=True).nda.dtype == np.ubyte

    sig_in_dec = google.decode(voev, zigzag=True)
    np.array_equal(sig_in_dec, wftable.values)
