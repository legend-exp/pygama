import numpy as np
from scipy.stats import norm as scipy_gaussian

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.pygama_continuous import PygamaContinuous


def test_gaussian_pdf():
    x = np.arange(-10, 12)
    mu = 1
    sigma = 2

    y = gaussian.pdf(x, mu, sigma)
    y_direct = gaussian.get_pdf(x, mu, sigma)
    scipy_y = scipy_gaussian.pdf(x, mu, sigma)

    assert isinstance(gaussian, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    x_lo = -100
    x_hi = 100

    y_sig, y_ext = gaussian.pdf_ext(x, x_lo, x_hi, n_sig, mu, sigma)
    assert np.allclose(y_ext, 20 * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-8)

    normalization = np.diff(scipy_gaussian.cdf(x[np.array([0, -1])], mu, sigma))
    y_norm = gaussian.pdf_norm(x, x[0], x[-1], mu, sigma)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_gaussian_cdf():
    x = np.arange(-10, 12)
    mu = 1
    sigma = 2

    y = gaussian.cdf(x, mu, sigma)
    y_direct = gaussian.get_cdf(x, mu, sigma)
    scipy_y = scipy_gaussian.cdf(x, mu, sigma)

    assert isinstance(gaussian, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = gaussian.cdf_ext(x, n_sig, mu, sigma)
    assert np.allclose(y_ext, 20 * scipy_y, rtol=1e-8)

    normalization = np.diff(scipy_gaussian.cdf(x[np.array([0, -1])], mu, sigma))
    y_norm = gaussian.cdf_norm(x, x[0], x[-1], mu, sigma)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_required_args():
    names = gaussian.required_args()
    assert names[0] == "mu"
    assert names[1] == "sigma"


def test_name():
    assert gaussian.name == "gaussian"
