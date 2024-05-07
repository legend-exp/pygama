import numpy as np
from scipy.stats import norm

from pygama.math.functions.gauss_on_linear import gauss_on_linear
from pygama.math.functions.sum_dists import SumDists


def line(x, m, b):
    return m * x + b


def quad(x, x_lo, m, b):
    return m * (x**2 - x_lo**2) / 2 + b * (x - x_lo)


def test_gauss_on_linear_pdf():
    x = np.arange(-10, 10)
    mu = 1
    sigma = 2
    x_lo = np.amin(x)
    x_hi = np.amax(x)
    m = 2
    b = 3
    n_sig = 10
    n_bkg = 20

    pars = np.array([x_lo, x_hi, n_sig, mu, sigma, n_bkg, m, b], dtype=float)

    assert isinstance(gauss_on_linear, SumDists)

    y_direct = gauss_on_linear.get_pdf(x, *pars)

    normalization = m / 2 * (x_hi**2 - x_lo**2) + b * (x_hi - x_lo)
    scipy_linear = n_bkg * line(x, m, b) / normalization

    scipy_gauss = n_sig * norm.pdf(x, mu, sigma)

    scipy_y = scipy_gauss + scipy_linear

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    x_lo = -10
    x_hi = 9

    y_sig, y_ext = gauss_on_linear.pdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig + n_bkg, rtol=1e-3)


def test_gauss_on_linear_cdf():
    x = np.arange(-10, 10)
    mu = 1
    sigma = 2
    x_lo = np.amin(x)
    x_hi = np.amax(x)
    m = 2
    b = 3
    n_sig = 10
    n_bkg = 20

    pars = np.array([x_lo, x_hi, n_sig, mu, sigma, n_bkg, m, b], dtype=float)

    assert isinstance(gauss_on_linear, SumDists)

    y_direct = gauss_on_linear.get_cdf(x, *pars)

    scipy_gauss = n_sig * norm.cdf(x, mu, sigma)
    normalization = m / 2 * (x_hi**2 - x_lo**2) + b * (x_hi - x_lo)
    scipy_linear = n_bkg * quad(x, x_lo, m, b) / normalization

    scipy_y = scipy_gauss + scipy_linear

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_ext = gauss_on_linear.cdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)


def test_required_args():
    names = gauss_on_linear.required_args()
    assert names[0] == "x_lo"
    assert names[1] == "x_hi"
    assert names[2] == "n_sig"
    assert names[3] == "mu"
    assert names[4] == "sigma"
    assert names[5] == "n_bkg"
    assert names[6] == "m"
    assert names[7] == "b"


def test_name():
    assert gauss_on_linear.name == "gauss_on_linear"
