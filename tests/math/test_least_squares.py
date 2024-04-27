import numpy as np

import pygama.math.least_squares as pgls


def test_linear_fit_by_sums():
    x = np.arange(0, 10)
    y = 2 * x + 1
    m, b = pgls.linear_fit_by_sums(x, y)

    assert m == 2
    assert b == 1


def test_fit_simple_scaling():
    x = np.arange(0, 10)
    y = 2 * x + 1
    m, b = pgls.fit_simple_scaling(x, y)

    assert m == 2.1578947368421053
    assert b == 0.0035087719298245615
