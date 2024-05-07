import numpy as np
from scipy.stats import moyal as scipy_moyal

from pygama.math.functions.moyal import moyal
from pygama.math.functions.pygama_continuous import PygamaContinuous


def test_moyal_pdf():
    x = np.arange(-10, 12)
    mu = 3
    sigma = 3

    y = moyal.pdf(x, mu, sigma)
    y_direct = moyal.get_pdf(x, mu, sigma)
    scipy_y = scipy_moyal.pdf(x, mu, sigma)

    assert isinstance(moyal, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    x_lo = -10
    x_hi = 50

    y_sig, y_ext = moyal.pdf_ext(x, x_lo, x_hi, n_sig, mu, sigma)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-3)

    normalization = np.diff(scipy_moyal.cdf(x[np.array([0, -1])], mu, sigma))
    y_norm = moyal.pdf_norm(x, x[0], x[-1], mu, sigma)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_moyal_cdf():
    x = np.arange(-10, 12)
    mu = 1
    sigma = 20

    y = moyal.cdf(x, mu, sigma)
    y_direct = moyal.get_cdf(x, mu, sigma)
    scipy_y = scipy_moyal.cdf(x, mu, sigma)

    assert isinstance(moyal, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = moyal.cdf_ext(x, n_sig, mu, sigma)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)

    normalization = np.diff(scipy_moyal.cdf(x[np.array([0, -1])], mu, sigma))
    y_norm = moyal.cdf_norm(x, x[0], x[-1], mu, sigma)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_required_args():
    names = moyal.required_args()
    assert names[0] == "mu"
    assert names[1] == "sigma"


def test_name():
    assert moyal.name == "moyal"
