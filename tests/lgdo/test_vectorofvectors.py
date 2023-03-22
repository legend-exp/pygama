import numpy as np
import pytest

from pygama import lgdo
from pygama.lgdo import VectorOfVectors


@pytest.fixture()
def lgdo_vov():
    return VectorOfVectors(
        flattened_data=lgdo.Array(
            nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])
        ),
        cumulative_length=lgdo.Array(nda=np.array([2, 5, 6, 10, 13])),
    )
    # [1 2]
    # [3 4 5]
    # [2]
    # [4 8 9 7]
    # [5 3 1]


def test_init(lgdo_vov):
    assert len(VectorOfVectors(dtype="ubyte")) == 0

    vov = VectorOfVectors(shape_guess=(10, 20), dtype="int32", fill_val=2)
    assert vov.flattened_data == lgdo.Array(shape=(10 * 20,), fill_val=2, dtype="int32")
    assert vov.cumulative_length == lgdo.Array(
        np.arange(20, 10 * 20 + 1, 20, dtype="uint32")
    )

    test = VectorOfVectors([[1, 2], [3, 4, 5], [2], [4, 8, 9, 7], [5, 3, 1]])
    assert test == lgdo_vov
    assert len(test) == 5

    vov = VectorOfVectors(
        cumulative_length=np.array([5, 10, 15], dtype="uint32"), dtype="ubyte"
    )
    assert len(vov.flattened_data) == 15
    assert len(vov[-1]) == 5

    vov = VectorOfVectors(shape_guess=(5, 0), dtype="int32")
    assert vov.cumulative_length == lgdo.Array([0, 0, 0, 0, 0])


def test_datatype_name(lgdo_vov):
    assert lgdo_vov.datatype_name() == "array"


def test_form_datatype(lgdo_vov):
    assert lgdo_vov.form_datatype() == "array<1>{array<1>{real}}"


def test_values(lgdo_vov):
    desired = [
        np.array([1, 2]),
        np.array([3, 4, 5]),
        np.array([2]),
        np.array([4, 8, 9, 7]),
        np.array([5, 3, 1]),
    ]

    for i in range(len(desired)):
        assert np.array_equal(desired[i], lgdo_vov[i])

    assert np.array_equal(lgdo_vov[-1], desired[-1])
    assert np.array_equal(lgdo_vov[-2], desired[-2])

    vov = VectorOfVectors([[1, 2]], dtype="uint32")
    assert np.array_equal(vov[-1], [1, 2])


def test_resize(lgdo_vov):
    lgdo_vov.resize(3)
    assert len(lgdo_vov.cumulative_length) == 3
    assert len(lgdo_vov.flattened_data) == lgdo_vov.cumulative_length[-1]

    desired = [np.array([1, 2]), np.array([3, 4, 5]), np.array([2])]

    for i in range(len(lgdo_vov)):
        assert np.array_equal(desired[i], lgdo_vov[i])

    lgdo_vov.resize(5)
    assert len(lgdo_vov) == 5
    assert len(lgdo_vov[3]) == 0
    assert len(lgdo_vov[4]) == 0
    assert lgdo_vov == VectorOfVectors([[1, 2], [3, 4, 5], [2], [], []])

    vov = VectorOfVectors(dtype="i")
    vov.resize(3)
    assert vov == VectorOfVectors([[], [], []], dtype="i")


def test_aoesa(lgdo_vov):
    arr = lgdo_vov.to_aoesa()
    desired = np.array(
        [
            [1, 2, np.nan, np.nan],
            [3, 4, 5, np.nan],
            [2, np.nan, np.nan, np.nan],
            [4, 8, 9, 7],
            [5, 3, 1, np.nan],
        ]
    )
    assert isinstance(arr, lgdo.ArrayOfEqualSizedArrays)
    assert arr.dtype == np.float64
    assert np.array_equal(arr.nda, desired, True)

    vov = VectorOfVectors(
        flattened_data=lgdo.Array(
            nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1], dtype="int16")
        ),
        cumulative_length=lgdo.Array(nda=np.array([2, 5, 6, 10, 13])),
    )
    aoesa = vov.to_aoesa()
    assert aoesa.dtype == np.float64

    aoesa = vov.to_aoesa(preserve_dtype=True)
    assert aoesa.dtype == np.int16


def test_set_vector(lgdo_vov):
    lgdo_vov[0] = np.zeros(2)

    desired = [
        np.zeros(2),
        np.array([3, 4, 5]),
        np.array([2]),
        np.array([4, 8, 9, 7]),
        np.array([5, 3, 1]),
    ]

    for i in range(len(desired)):
        assert np.array_equal(desired[i], lgdo_vov[i])

    with pytest.raises(ValueError):
        lgdo_vov[0] = np.zeros(3)

    lgdo_vov[1] = np.zeros(3)

    desired = [
        np.zeros(2),
        np.zeros(3),
        np.array([2]),
        np.array([4, 8, 9, 7]),
        np.array([5, 3, 1]),
    ]

    for i in range(len(desired)):
        assert np.array_equal(desired[i], lgdo_vov[i])


def test_append(lgdo_vov):
    lgdo_vov.append(np.zeros(3))
    assert np.array_equal(lgdo_vov[-1], np.zeros(3))

    vov = VectorOfVectors(dtype="int64")
    vov.append(np.zeros(3))
    assert vov == VectorOfVectors([[0, 0, 0]])


def test_insert(lgdo_vov):
    lgdo_vov.insert(2, np.zeros(3))
    assert lgdo_vov == VectorOfVectors(
        [
            [1, 2],
            [3, 4, 5],
            [0, 0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )

    vov = VectorOfVectors(shape_guess=(3, 5), dtype="int32", fill_val=0)
    vov.insert(2, [1, 2, 3])
    assert np.array_equal(vov.cumulative_length, [5, 10, 13, 18])
    assert np.array_equal(vov[2], [1, 2, 3])


def test_replace(lgdo_vov):
    vov = lgdo.copy(lgdo_vov)
    vov.replace(1, np.zeros(3))
    assert vov == VectorOfVectors(
        [
            [1, 2],
            [0, 0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )

    vov = lgdo.copy(lgdo_vov)
    vov.replace(1, np.zeros(2))
    assert vov == VectorOfVectors(
        [
            [1, 2],
            [0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )

    vov = lgdo.copy(lgdo_vov)
    vov.replace(1, np.zeros(4))
    assert vov == VectorOfVectors(
        [
            [1, 2],
            [0, 0, 0, 0],
            [2],
            [4, 8, 9, 7],
            [5, 3, 1],
        ]
    )


def test_iter(lgdo_vov):
    desired = [
        np.array([1, 2]),
        np.array([3, 4, 5]),
        np.array([2]),
        np.array([4, 8, 9, 7]),
        np.array([5, 3, 1]),
    ]

    c = 0
    for v in lgdo_vov:
        assert (v == desired[c]).all()
        c += 1


def test_build_cl_and_explodes():
    cl = np.array([3, 4], dtype=np.uint64)
    exp = np.array([0, 0, 0, 1], dtype=np.uint64)
    array = np.array([5, 7], dtype=np.uint64)
    array_exp = np.array([5, 5, 5, 7], dtype=np.uint64)
    # build_cl
    assert (lgdo.build_cl(exp, cl) == cl).all()
    assert (lgdo.build_cl(exp) == cl).all()
    assert (lgdo.build_cl([0, 0, 0, 1]) == cl).all()
    assert (lgdo.build_cl(array_exp, cl) == cl).all()
    assert (lgdo.build_cl(array_exp) == cl).all()
    assert (lgdo.build_cl([5, 5, 5, 7]) == cl).all()
    # explode_cl
    assert (lgdo.explode_cl(cl, exp) == exp).all()
    assert (lgdo.explode_cl(cl) == exp).all()
    assert (lgdo.explode_cl([3, 4]) == exp).all()
    # inverse functionality
    assert (lgdo.build_cl(lgdo.explode_cl(cl)) == cl).all()
    assert (lgdo.explode_cl(lgdo.build_cl(array_exp)) == exp).all()
    # explode
    assert (lgdo.explode(cl, array, array_exp) == array_exp).all()
    assert (lgdo.explode(cl, array) == array_exp).all()
    assert (lgdo.explode([3, 4], [5, 7]) == array_exp).all()
    assert (lgdo.explode(cl, range(len(cl))) == exp).all()
    # explode_arrays
    arrays_out = lgdo.explode_arrays(cl, [array, range(len(cl))])
    assert len(arrays_out) == 2
    assert (arrays_out[0] == array_exp).all()
    assert (arrays_out[1] == exp).all()
    arrays_out = lgdo.explode_arrays(cl, [array, range(len(cl))], arrays_out=arrays_out)
    assert len(arrays_out) == 2
    assert (arrays_out[0] == array_exp).all()
    assert (arrays_out[1] == exp).all()


def test_copy(lgdo_vov):
    assert lgdo_vov == lgdo.copy(lgdo_vov)
