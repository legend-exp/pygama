import numpy as np
from scipy.stats import exponnorm, norm

from pygama.math.functions.gauss_on_exgauss import gauss_on_exgauss
from pygama.math.functions.sum_dists import sum_dists


def test_gauss_on_exgauss_pdf():

    x = np.arange(-10, 10)
    mu = 1
    sigma = 1
    tau = 3
    h_tail = 0.25

    pars = np.array([mu, sigma, h_tail, tau], dtype=float)

    assert isinstance(gauss_on_exgauss, sum_dists)

    y_direct = gauss_on_exgauss.get_pdf(x, pars)
    scipy_exgauss = h_tail * exponnorm.pdf(
        -1 * x, tau / sigma, -1 * mu, sigma
    )  # to be equivalent to the scipy version, x -> -x, mu -> -mu, k -> k/sigma
    scipy_gauss = (1 - h_tail) * norm.pdf(x, mu, sigma)

    scipy_y = scipy_exgauss + scipy_gauss

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    x_lo = -100
    x_hi = 100
    pars = np.insert(pars, 0, [x_lo, x_hi])
    y_sig, y_ext = gauss_on_exgauss.pdf_ext(x, pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, 1, rtol=1e-8)


def test_gauss_on_exgauss_cdf():

    x = np.arange(-10, 10)
    mu = 1
    sigma = 1
    tau = 3

    h_tail = 0.25

    pars = np.array([mu, sigma, h_tail, tau], dtype=float)

    assert isinstance(gauss_on_exgauss, sum_dists)

    y_direct = gauss_on_exgauss.get_cdf(x, pars)

    scipy_exgauss = h_tail * (
        1 - exponnorm.cdf(-1 * x, tau / sigma, -1 * mu, sigma)
    )  # to be equivalent to the scipy version, x -> -x, mu -> -mu, k -> k/sigma
    scipy_gauss = (1 - h_tail) * norm.cdf(x, mu, sigma)

    scipy_y = scipy_exgauss + scipy_gauss

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_ext = gauss_on_exgauss.cdf_ext(x, pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
