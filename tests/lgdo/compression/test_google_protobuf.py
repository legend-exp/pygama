import numpy as np

from pygama.lgdo.compression import google


def _to_bin(array):
    return [np.binary_repr(x, width=8) for x in array]


def test_zigzag():
    assert google.zigzag_encode(0) == 0
    assert google.zigzag_encode(-1) == 1
    assert google.zigzag_encode(1) == 2
    assert google.zigzag_encode(-2) == 3
    assert google.zigzag_encode(2) == 4

    assert google.zigzag_decode(0) == 0
    assert google.zigzag_decode(1) == -1
    assert google.zigzag_decode(2) == 1
    assert google.zigzag_decode(3) == -2
    assert google.zigzag_decode(4) == 2


def test_varint_encoding():
    encx = np.empty(10, dtype="ubyte")
    pos = google.unsigned_varint_encode(1, encx)
    assert _to_bin(encx[0:pos]) == ["00000001"]
    assert google.unsigned_varint_decode(encx[0:pos]) == (1, 1)

    pos = google.unsigned_varint_encode(150, encx)
    assert _to_bin(encx[0:pos]) == ["10010110", "00000001"]
    assert google.unsigned_varint_decode(encx[0:pos]) == (150, 2)


def test_varint_encode_decode_equality():
    for x in [1, 3856, 234, 11, 93645, 2, 45]:
        encx = np.empty(100, dtype="ubyte")
        pos = google.unsigned_varint_encode(x, encx)
        encx = np.resize(encx, pos)
        assert google.unsigned_varint_decode(encx)[0] == x


def test_varint_array_encode_decode_equality():
    sig_in = np.array([1, 3856, 234, 11, 93645, 2, 45], dtype="ubyte")
    sig_out = np.empty(100, dtype="ubyte")

    google.unsigned_varint_array_encode(sig_in, sig_out)

    sig_in_dec = np.empty(100, dtype="uint32")
    google.unsigned_varint_array_decode(sig_out, sig_in_dec)
    assert np.array_equal(sig_in, sig_in_dec[0:7])


def test_zigzag_varint_array_encode_decode_equality():
    sig_in = np.array([1, -3856, -234, 11, 93645, -2, 45])
    sig_out = np.empty(100, dtype="ubyte")

    google.unsigned_varint_array_encode(
        google.zigzag_encode(sig_in).astype("uint32"), sig_out
    )

    sig_in_dec = np.empty(100, dtype="uint32")
    google.unsigned_varint_array_decode(sig_out, sig_in_dec)
    assert np.array_equal(sig_in, google.zigzag_decode(sig_in_dec[0:7]))
