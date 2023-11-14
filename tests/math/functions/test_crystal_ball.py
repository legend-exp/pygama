import numpy as np
from scipy.stats import crystalball as scipy_crystal_ball

from pygama.math.functions.crystal_ball import crystal_ball
from pygama.math.functions.pygama_continuous import pygama_continuous


def test_crystalball_pdf():

    x = np.arange(-10, 12)
    beta = 2
    m = 3
    mu = 1
    sigma = 2

    y = crystal_ball.pdf(x, beta, m, mu, sigma)
    y_direct = crystal_ball.get_pdf(x, beta, m, mu, sigma)
    scipy_y = scipy_crystal_ball.pdf(x, beta, m, mu, sigma)

    assert isinstance(crystal_ball, pygama_continuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    x_lo = -10000
    x_hi = 10000
    crystal_ball.set_x_lo(x_lo)
    crystal_ball.set_x_hi(x_hi)

    y_sig, y_ext = crystal_ball.pdf_ext(x, n_sig, beta, m, mu, sigma)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-8)

    crystal_ball.set_x_lo(x[0])
    crystal_ball.set_x_hi(x[-1])

    normalization = np.diff(
        scipy_crystal_ball.cdf(x[np.array([0, -1])], beta, m, mu, sigma)
    )
    y_norm = crystal_ball.pdf_norm(x, beta, m, mu, sigma)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_crystalball_cdf():

    x = np.arange(-10, 12)
    beta = 2
    m = 3
    mu = 1
    sigma = 2

    y = crystal_ball.cdf(x, beta, m, mu, sigma)
    y_direct = crystal_ball.get_cdf(x, beta, m, mu, sigma)
    scipy_y = scipy_crystal_ball.cdf(x, beta, m, mu, sigma)

    assert isinstance(crystal_ball, pygama_continuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = crystal_ball.cdf_ext(x, n_sig, beta, m, mu, sigma)
    assert np.allclose(y_ext, 20 * scipy_y, rtol=1e-8)

    crystal_ball.set_x_lo(x[0])
    crystal_ball.set_x_hi(x[-1])

    normalization = np.diff(
        scipy_crystal_ball.cdf(x[np.array([0, -1])], beta, m, mu, sigma)
    )
    y_norm = crystal_ball.cdf_norm(x, beta, m, mu, sigma)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)
