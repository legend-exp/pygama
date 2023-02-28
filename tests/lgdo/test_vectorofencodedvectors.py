import numpy as np
import pytest

from pygama.lgdo import Array, VectorOfEncodedVectors, VectorOfVectors


def test_init():
    with pytest.raises(ValueError):
        VectorOfEncodedVectors()

    voev = VectorOfEncodedVectors(
        VectorOfVectors(shape_guess=(100, 1000), dtype="uint16")
    )
    assert len(voev.decoded_size) == 100
    assert voev.attrs["datatype"] == "array<1>{encoded_array<1>{real}}"
    assert len(voev) == 100

    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(shape_guess=(100, 1000), dtype="uint16"),
        decoded_size=Array(shape=100),
        attrs={"sth": 1},
    )
    assert voev.attrs == {"datatype": "array<1>{encoded_array<1>{real}}", "sth": 1}


def test_resize():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(shape_guess=(100, 1000), dtype="uint16"),
        decoded_size=Array(shape=100),
        attrs={"sth": 1},
    )
    voev.resize(50)
    assert len(voev) == 50


def test_set_get_vector():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(
            shape_guess=(100, 1000), dtype="uint16", fill_val=0
        ),
        decoded_size=Array(shape=100),
        attrs={"sth": 1},
    )
    voev.set_vector(5, np.array([1, 2, 3]), 7)
    assert (voev[5][0] == np.array([1, 2, 3])).all()
    assert voev[5][1] == 7

    assert (voev[5][0] == np.array([1, 2, 3])).all()
    assert voev[5][1] == 7

    voev[6] = (np.array([1, 2, 3]), 7)
    assert (voev[6][0] == np.array([1, 2, 3])).all()
    assert voev[6][1] == 7


def test_iteration():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(
            flattened_data=Array(nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])),
            cumulative_length=Array(nda=np.array([2, 5, 6, 10, 13])),
        ),
        decoded_size=Array(shape=5, fill_val=6),
    )

    desired = [
        np.array([1, 2]),
        np.array([3, 4, 5]),
        np.array([2]),
        np.array([4, 8, 9, 7]),
        np.array([5, 3, 1]),
    ]

    for i, (v, s) in enumerate(voev):
        assert (v == desired[i]).all()
        assert s == 6
