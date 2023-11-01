import numpy as np
from scipy.stats import uniform as scipy_uniform

from pygama.math.functions.linear import linear
from pygama.math.functions.pygama_continuous import pygama_continuous
from pygama.math.functions.uniform import uniform


def test_linear_pdf():

    x = np.arange(-10, 12)
    m = 0
    b = 2
    x_lo = np.inf
    x_hi = np.inf

    y = linear.pdf(x, x_lo, x_hi, m, b)
    y_direct = linear.get_pdf(x, x_lo, x_hi, m, b)
    scipy_y = scipy_uniform.pdf(x, x[0], x[-1] + np.abs(x[0]))

    assert isinstance(linear, pygama_continuous)
    assert np.array_equal(y, scipy_y)
    assert np.array_equal(y_direct, scipy_y)

    n_sig = 20
    x_lo = -10.0
    x_hi = 12.0
    y_sig, y_ext = linear.pdf_ext(x, n_sig, x_lo, x_hi, np.inf, np.inf, m, b)
    assert np.array_equal(y_ext, n_sig * scipy_y)
    assert np.array_equal(y_sig, n_sig)

    norm_y = linear.pdf_norm(x, x[0], x[-1], m, b)
    assert np.array_equal(norm_y, scipy_y)


def test_linear_cdf():

    x = np.arange(-10, 12)
    m = 0
    b = 2
    x_lo = np.inf
    x_hi = np.inf

    y = linear.cdf(x, x_lo, x_hi, m, b)
    y_direct = linear.get_cdf(x, x_lo, x_hi, m, b)
    scipy_y = scipy_uniform.cdf(x, x[0], x[-1] + np.abs(x[0]))

    assert isinstance(uniform, pygama_continuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = linear.cdf_ext(x, n_sig, np.inf, np.inf, m, b)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)

    norm_y = linear.cdf_norm(x, x[0], x[-1], m, b)
    assert np.allclose(norm_y, scipy_y)
