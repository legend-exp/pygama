import numpy as np
import pytest

import pygama.lgdo as lgdo


@pytest.fixture()
def lgdo_vov():
    return lgdo.VectorOfVectors(
        flattened_data=lgdo.Array(nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1])),
        cumulative_length=lgdo.Array(nda=np.array([2, 5, 6, 10, 13]))
    )


def test_init(lgdo_vov):
    pass


def test_datatype_name(lgdo_vov):
    assert lgdo_vov.datatype_name() == 'array'


def test_form_datatype(lgdo_vov):
    assert lgdo_vov.form_datatype() == 'array<1>{array<1>{real}}'


def test_values(lgdo_vov):
    desired = [np.array([1, 2]),
               np.array([3, 4, 5]),
               np.array([2]),
               np.array([4, 8, 9, 7]),
               np.array([5, 3, 1])]

    for i in range(len(desired)):
        assert (desired[i] == list(lgdo_vov)[i]).all()


def test_resize(lgdo_vov):
    lgdo_vov.resize(3)
    assert len(lgdo_vov.cumulative_length) == 3


def test_set_vector(lgdo_vov):
    lgdo_vov.set_vector(0, np.zeros(2))

    desired = [np.zeros(2),
               np.array([3, 4, 5]),
               np.array([2]),
               np.array([4, 8, 9, 7]),
               np.array([5, 3, 1])]

    for i in range(len(desired)):
        assert (desired[i] == list(lgdo_vov)[i]).all()


def test_iter(lgdo_vov):
    desired = [np.array([1, 2]),
               np.array([3, 4, 5]),
               np.array([2]),
               np.array([4, 8, 9, 7]),
               np.array([5, 3, 1])]

    c = 0
    for v in lgdo_vov:
        assert (v == desired[c]).all()
        c += 1
