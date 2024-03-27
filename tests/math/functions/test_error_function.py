from math import erf, erfc

import numpy as np

import pygama.math.functions.error_function as pgfef


def test_erf():
    x = np.arange(-10, 12)

    y = pgfef.nb_erf(x)
    y_math = []
    for i in x:
        y_math.append(erf(i))
    assert np.array_equal(y, y_math)


def test_erfc():
    x = np.arange(-10, 12)

    y = pgfef.nb_erfc(x)
    y_math = []
    for i in x:
        y_math.append(erfc(i))
    assert np.array_equal(y, y_math)
