import pytest
from numpy.testing import assert_
import numpy as np

import pygama.lgdo as lgdo


@pytest.fixture()
def vov():
    print('--------setup--------')
    yield lgdo.VectorOfVectors()
    print('--------tear down--------')

@pytest.fixture()
def array():
    print('--------setup--------')
    yield lgdo.Array

class Test_VectorOfVectors:

    def test_datatype_name(self, vov):
        result = vov.datatype_name()
        desired = 'array'
        assert_(result == desired)


    def test_list(self, vov):
        flat = array(np.array([1, 2, 3, 4, 5, 2, 4, 8, 9, 7, 5, 3, 1]))
        length = array(np.array([2, 5, 6, 10, 13]))
        v = vov(flattened_data=flat, cumulative_length=length)
        result = list(vov)

        desired = [np.array(v.flattened_data.nda[:v.cumulative_length.nda[0]])]
        desired += [np.array(v.flattened_data.nda[v.cumulative_length.nda[i]:v.cumulative_length.nda[i+1]]) for i in range(len(v.cumulative_length))]

        assert_(desired==result)
