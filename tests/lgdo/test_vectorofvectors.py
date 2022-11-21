import numpy as np
import pytest

import pygama.lgdo as lgdo


@pytest.fixture()
def lgdo_vov():
    return lgdo.VectorOfVectors(
        flattened_data=lgdo.Array(
            nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])
        ),
        cumulative_length=lgdo.Array(nda=np.array([2, 5, 6, 10, 13])),
    )


def test_init(lgdo_vov):
    pass


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
        assert (desired[i] == list(lgdo_vov)[i]).all()


def test_resize(lgdo_vov):
    lgdo_vov.resize(3)
    assert len(lgdo_vov.cumulative_length) == 3


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
    assert np.array_equal(arr.nda, desired, True)


def test_set_vector(lgdo_vov):
    lgdo_vov.set_vector(0, np.zeros(2))

    desired = [
        np.zeros(2),
        np.array([3, 4, 5]),
        np.array([2]),
        np.array([4, 8, 9, 7]),
        np.array([5, 3, 1]),
    ]

    for i in range(len(desired)):
        assert (desired[i] == list(lgdo_vov)[i]).all()


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
    out_arrays = lgdo.explode_arrays(cl, [array, range(len(cl))])
    assert len(out_arrays) == 2
    assert (out_arrays[0] == array_exp).all()
    assert (out_arrays[1] == exp).all()
    out_arrays = lgdo.explode_arrays(cl, [array, range(len(cl))], out_arrays=out_arrays)
    assert len(out_arrays) == 2
    assert (out_arrays[0] == array_exp).all()
    assert (out_arrays[1] == exp).all()
