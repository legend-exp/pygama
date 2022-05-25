import numpy as np
import pytest
from numpy.testing import assert_

import pygama.lgdo as lgdo


@pytest.fixture()
def vov():
    print('--------setup--------')

    yield lgdo.VectorOfVectors
    print('--------tear down--------')

@pytest.fixture()
def array():
    print('--------setup--------')
    yield lgdo.Array


class Test_VectorOfVectors:
    def test_datatype_name(self, vov, array):
        flat = array(nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1]))
        length = array(nda=np.array([2, 5, 6, 10, 13]))
        v = vov(flattened_data=flat, cumulative_length=length)
        result = v.datatype_name()
        desired = 'array'
        assert_(result == desired)


    def test_list(self, vov, array):
        flat = array(nda=np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1]))
        length = array(nda=np.array([2, 5, 6, 10, 13]))
        v = vov(flattened_data=flat, cumulative_length=length)
        result = list(v)

        desired = [ np.array([1,2]),
                    np.array([3,4,5]),
                    np.array([2]),
                    np.array([4,8,9,7]),
                    np.array([5, 3, 1]) ]

        assert_( (desired[i]==result[i]).all() for i in range(len(desired)) )
