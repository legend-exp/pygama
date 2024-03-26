import numpy as np
from numpy.polynomial.polynomial import Polynomial

from pygama.math.functions.polynomial import nb_poly


def test_nb_poly():
    x = np.arange(-10, 10)
    params = np.array([1, 2, 3])
    y = nb_poly(x, params)
    np_poly = Polynomial(params)
    y_numpy = np_poly(x)

    assert np.array_equal(y, y_numpy)
