import numpy as np
from scipy.stats import uniform as scipy_uniform

from pygama.math.functions.pygama_continuous import pygama_continuous
from pygama.math.functions.uniform import uniform


def test_uniform_pdf():

    x = np.arange(-10, 12)
    a = -1
    b = 10

    y = uniform.pdf(x, a, b)
    y_direct = uniform.get_pdf(x, a, b)
    scipy_y = scipy_uniform.pdf(x, a, b)

    assert isinstance(uniform, pygama_continuous)
    assert np.array_equal(y, scipy_y)
    assert np.array_equal(y_direct, scipy_y)

    n_sig = 20
    x_lo = -1
    x_hi = 9
    y_sig, y_ext = uniform.pdf_ext(x, n_sig, x_lo, x_hi, a, b)
    assert np.array_equal(y_ext, 20 * scipy_y)
    assert np.array_equal(y_sig, 20)

    y_norm = uniform.norm_pdf(x, x[0], x[-1], a, b)

    assert np.allclose(y_norm, y_direct, rtol=1e-8)


def test_uniform_cdf():

    x = np.arange(-10, 12)
    a = -1
    b = 10

    y = uniform.cdf(x, a, b)
    y_direct = uniform.get_cdf(x, a, b)
    scipy_y = scipy_uniform.cdf(x, a, b)

    assert isinstance(uniform, pygama_continuous)
    assert np.array_equal(y, scipy_y)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = uniform.cdf_ext(x, n_sig, a, b)
    assert np.allclose(y_ext, 20 * scipy_y, rtol=1e-8)

    y_norm = uniform.norm_cdf(x, x[0], x[-1], a, b)

    assert np.allclose(y_norm, y_direct, rtol=1e-8)
