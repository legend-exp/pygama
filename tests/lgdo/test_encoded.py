import numpy as np

from pygama.lgdo import (
    Array,
    ArrayOfEncodedEqualSizedArrays,
    Scalar,
    VectorOfEncodedVectors,
    VectorOfVectors,
)


def test_voev_init():
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


def test_aoeesa_init():
    voev = ArrayOfEncodedEqualSizedArrays(
        VectorOfVectors(shape_guess=(100, 1000), dtype="uint16")
    )
    assert isinstance(voev.decoded_size, Scalar)
    assert voev.attrs["datatype"] == "array_of_encoded_equalsized_arrays<1,1>{real}"
    assert len(voev) == 100

    voev = ArrayOfEncodedEqualSizedArrays(
        encoded_data=VectorOfVectors(shape_guess=(100, 1000), dtype="uint16"),
        decoded_size=99,
        attrs={"sth": 1},
    )
    assert voev.decoded_size.value == 99
    assert voev.attrs == {
        "datatype": "array_of_encoded_equalsized_arrays<1,1>{real}",
        "sth": 1,
    }


def test_resize():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(shape_guess=(100, 1000), dtype="uint16"),
        decoded_size=Array(shape=100),
    )
    voev.resize(50)
    assert len(voev) == 50

    voev = ArrayOfEncodedEqualSizedArrays(
        encoded_data=VectorOfVectors(shape_guess=(100, 1000), dtype="uint16"),
        decoded_size=99,
    )
    voev.resize(50)
    assert len(voev) == 50


def test_append():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(shape_guess=(100, 1000), dtype="uint16"),
        decoded_size=Array(shape=100),
    )
    voev.append(([1, 3, 5], 99))
    assert len(voev) == 101
    assert (voev[-1][0] == [1, 3, 5]).all()
    assert voev[-1][1] == 99

    voev = ArrayOfEncodedEqualSizedArrays(
        encoded_data=VectorOfVectors(shape_guess=(100, 1000), dtype="uint16"),
        decoded_size=100,
    )
    voev.append([1, 3, 5])
    assert len(voev) == 101
    assert (voev[-1] == [1, 3, 5]).all()
    assert voev.decoded_size.value == 100


def test_insert():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(
            shape_guess=(100, 1000), dtype="uint16", fill_val=0
        ),
        decoded_size=Array(shape=100, fill_val=12),
    )
    voev.insert(3, ([1, 3, 5], 99))
    assert len(voev) == 101
    assert (voev[3][0] == [1, 3, 5]).all()
    assert voev[3][1] == 99

    voev = ArrayOfEncodedEqualSizedArrays(
        encoded_data=VectorOfVectors(
            shape_guess=(100, 1000), dtype="uint16", fill_val=0
        ),
        decoded_size=100,
    )

    voev.insert(4, [1, 3, 5])
    assert len(voev) == 101
    assert (voev[4] == [1, 3, 5]).all()
    assert voev.decoded_size.value == 100


def test_replace():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(
            shape_guess=(100, 1000), dtype="uint16", fill_val=0
        ),
        decoded_size=Array(shape=100, fill_val=12),
    )
    voev.replace(3, ([1, 3, 5], 99))
    assert len(voev) == 100
    assert (voev[3][0] == [1, 3, 5]).all()
    assert voev[3][1] == 99

    voev = ArrayOfEncodedEqualSizedArrays(
        encoded_data=VectorOfVectors(
            shape_guess=(100, 1000), dtype="uint16", fill_val=0
        ),
        decoded_size=100,
    )

    voev.replace(4, [1, 3, 5])
    assert len(voev) == 100
    assert (voev[4] == [1, 3, 5]).all()
    assert voev.decoded_size.value == 100


def test_voev_set_get_vector():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(shape_guess=(100, 3), dtype="uint16", fill_val=0),
        decoded_size=Array(shape=100),
        attrs={"sth": 1},
    )
    voev[5] = (np.array([1, 2, 3]), 7)
    assert np.array_equal(voev[5][0], np.array([1, 2, 3]))
    assert voev[5][1] == 7

    assert np.array_equal(voev[5][0], np.array([1, 2, 3]))
    assert voev[5][1] == 7

    voev[6] = (np.array([1, 2, 3]), 7)
    assert np.array_equal(voev[6][0], np.array([1, 2, 3]))
    assert voev[6][1] == 7


def test_aoeesa_set_get_vector():
    voev = ArrayOfEncodedEqualSizedArrays(
        encoded_data=VectorOfVectors(shape_guess=(100, 3), dtype="uint16", fill_val=0),
        decoded_size=99,
        attrs={"sth": 1},
    )
    voev[5] = np.array([1, 2, 3])
    assert np.array_equal(voev[5], np.array([1, 2, 3]))

    assert np.array_equal(voev[5], np.array([1, 2, 3]))

    voev[6] = np.array([1, 2, 3])
    assert np.array_equal(voev[6], np.array([1, 2, 3]))

    assert voev.decoded_size.value == 99


def test_voev_iteration():
    voev = VectorOfEncodedVectors(
        encoded_data=VectorOfVectors(
            flattened_data=Array(nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])),
            cumulative_length=Array(nda=np.array([2, 5, 6, 10, 13])),
        ),
        decoded_size=Array(shape=5, fill_val=6),
    )

    desired = [
        [1, 2],
        [3, 4, 5],
        [2],
        [4, 8, 9, 7],
        [5, 3, 1],
    ]

    for i, (v, s) in enumerate(voev):
        assert np.array_equal(v, desired[i])
        assert s == 6


def test_aoeesa_iteration():
    voev = ArrayOfEncodedEqualSizedArrays(
        encoded_data=VectorOfVectors(
            flattened_data=Array(nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])),
            cumulative_length=Array(nda=np.array([2, 5, 6, 10, 13])),
        ),
        decoded_size=99,
    )

    desired = [
        [1, 2],
        [3, 4, 5],
        [2],
        [4, 8, 9, 7],
        [5, 3, 1],
    ]

    for i, v in enumerate(voev):
        assert np.array_equal(v, desired[i])
