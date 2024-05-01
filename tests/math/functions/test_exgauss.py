import numpy as np
from scipy.stats import exponnorm

from pygama.math.functions.exgauss import exgauss
from pygama.math.functions.pygama_continuous import PygamaContinuous


def test_exgauss_pdf():
    x = np.arange(-10, 10)
    mu = 0.5
    sigma = 2
    tau = 2

    par_array = [mu, sigma, tau]

    y = exgauss.pdf(x, sigma, tau, mu, sigma)
    y_direct = exgauss.get_pdf(x, *par_array)
    scipy_y = exponnorm.pdf(
        -1 * x, tau / sigma, -1 * mu, sigma
    )  # to be equivalent to the scipy version, x -> -x, mu -> -mu, k -> k/sigma

    assert isinstance(exgauss, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 10
    x_lo = -100
    x_hi = 100

    y_sig, y_ext = exgauss.pdf_ext(x, x_lo, x_hi, n_sig, *par_array)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-8)

    normalization = np.diff(
        1 - exponnorm.cdf(-1 * x[np.array([0, -1])], tau / sigma, -1 * mu, sigma)
    )
    y_norm = exgauss.pdf_norm(x, x[0], x[-1], *par_array)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_exgauss_cdf():
    x = np.arange(-10, 10)
    mu = 0.5
    sigma = 2
    tau = 2

    par_array = [mu, sigma, tau]

    y = exgauss.cdf(x, sigma, tau, mu, sigma)
    scipy_y = 1 - exponnorm.cdf(
        -1 * x, tau / sigma, -1 * mu, sigma
    )  # to be equivalent to the scipy version, x -> -x, mu -> -mu, k -> k/sigma
    y_direct = exgauss.get_cdf(x, *par_array)

    assert isinstance(exgauss, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = exgauss.cdf_ext(x, n_sig, *par_array)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)

    normalization = np.diff(scipy_y[np.array([0, -1])])
    y_norm = exgauss.cdf_norm(x, x[0], x[-1], *par_array)
    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_required_args():
    names = exgauss.required_args()
    assert names[0] == "mu"
    assert names[1] == "sigma"
    assert names[2] == "tau"


def test_name():
    assert exgauss.name == "exgauss"
