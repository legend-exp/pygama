import numpy as np
import pygama.lgdo as lgdo


def test_datatype_name():
    v = lgdo.VectorOfVectors(shape_guess=(12, 64), dtype='uint8')
    assert v.datatype_name() == 'array'


def test_list():
    flat = lgdo.Array(nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1]))
    length = lgdo.Array(nda=np.array([2, 5, 6, 10, 13]))
    v = lgdo.VectorOfVectors(flattened_data=flat, cumulative_length=length)
    result = list(v)

    desired = [np.array([1, 2]),
               np.array([3, 4, 5]),
               np.array([2]),
               np.array([4, 8, 9, 7]),
               np.array([5, 3, 1])]

    for i in range(len(desired)):
        assert (desired[i] == result[i]).all()
