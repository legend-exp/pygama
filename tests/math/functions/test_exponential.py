import numpy as np
from scipy.stats import expon

from pygama.math.functions.exponential import exponential
from pygama.math.functions.pygama_continuous import PygamaContinuous


def test_exponential_pdf():
    x = np.arange(1, 12)
    mu = 3.1
    sigma = 2.3
    lamb = 0.2

    par_array = [mu, sigma, lamb]

    y = exponential.pdf(x, *par_array)
    y_direct = exponential.get_pdf(x, *par_array)
    scipy_y = expon.pdf(x, mu, sigma / lamb)

    assert isinstance(exponential, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    x_lo = 0
    x_hi = 200

    y_sig, y_ext = exponential.pdf_ext(x, x_lo, x_hi, n_sig, *par_array)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-3)

    normalization = np.diff(expon.cdf(x[np.array([0, -1])], mu, sigma / lamb))
    y_norm = exponential.pdf_norm(x, x[0], x[-1], *par_array)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_exponential_cdf():
    x = np.arange(5, 12)
    mu = 1.1
    sigma = 2.4
    lamb = 0.3
    par_array = [mu, sigma, lamb]

    y = exponential.cdf(x, *par_array)
    y_direct = exponential.get_cdf(x, *par_array)
    scipy_y = expon.cdf(x, mu, sigma / lamb)

    assert isinstance(exponential, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = exponential.cdf_ext(x, n_sig, *par_array)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)

    normalization = np.diff(expon.cdf(x[np.array([0, -1])], mu, sigma / lamb))
    y_norm = exponential.cdf_norm(x, x[0], x[-1], *par_array)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_required_args():
    names = exponential.required_args()
    assert names[0] == "mu"
    assert names[1] == "sigma"
    assert names[2] == "lambda"


def test_name():
    assert exponential.name == "exponential"
