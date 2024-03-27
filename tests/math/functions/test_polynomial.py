import numpy as np

from pygama.math.functions.polynomial import nb_poly


def test_nb_poly():
    x = np.arange(-10, 10)
    params = np.array([1, 2, 3])
    y = nb_poly(x, params)

    y_numpy = np.polyval(params, x)

    assert np.array_equal(y, y_numpy)
