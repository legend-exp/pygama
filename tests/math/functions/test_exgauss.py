import numpy as np
from scipy.stats import exponnorm

from pygama.math.functions.exgauss import exgauss
from pygama.math.functions.pygama_continuous import pygama_continuous


def test_exgauss_pdf():

    x = np.arange(-10, 10)
    mu = 0.5
    sigma = 2
    tau = 2

    y = exgauss.pdf(x, tau, sigma, mu, sigma)
    y_direct = exgauss.get_pdf(x, tau, mu, sigma)
    scipy_y = exponnorm.pdf(
        -1 * x, tau / sigma, -1 * mu, sigma
    )  # to be equivalent to the scipy version, x -> -x, mu -> -mu, k -> k/sigma

    assert isinstance(exgauss, pygama_continuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 10
    x_lo = -100
    x_hi = 100
    y_sig, y_ext = exgauss.pdf_ext(x, n_sig, x_lo, x_hi, tau, mu, sigma)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-8)

    normalization = np.diff(
        1 - exponnorm.cdf(-1 * x[np.array([0, -1])], tau / sigma, -1 * mu, sigma)
    )
    y_norm = exgauss.norm_pdf(x, x[0], x[-1], tau, mu, sigma)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_exgauss_cdf():

    x = np.arange(-10, 10)
    mu = 0.5
    sigma = 2
    tau = 2

    y = exgauss.cdf(x, tau, sigma, mu, sigma)
    scipy_y = 1 - exponnorm.cdf(
        -1 * x, tau / sigma, -1 * mu, sigma
    )  # to be equivalent to the scipy version, x -> -x, mu -> -mu, k -> k/sigma
    y_direct = exgauss.get_cdf(x, tau, mu, sigma)

    assert isinstance(exgauss, pygama_continuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = exgauss.cdf_ext(x, n_sig, tau, mu, sigma)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)

    normalization = np.diff(scipy_y[np.array([0, -1])])
    y_norm = exgauss.norm_cdf(x, x[0], x[-1], tau, mu, sigma)
    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)
