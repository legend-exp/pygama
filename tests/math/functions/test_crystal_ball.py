import numpy as np
from scipy.stats import crystalball as scipy_crystal_ball

from pygama.math.functions.crystal_ball import crystal_ball
from pygama.math.functions.pygama_continuous import PygamaContinuous


def test_crystalball_pdf():
    x = np.arange(-10, 12)
    beta = 2
    m = 3
    mu = 1
    sigma = 2

    par_array = [mu, sigma, beta, m]

    y = crystal_ball.pdf(x, *par_array)
    y_direct = crystal_ball.get_pdf(x, *par_array)
    scipy_y = scipy_crystal_ball.pdf(x, beta, m, mu, sigma)

    assert isinstance(crystal_ball, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    x_lo = -10000
    x_hi = 10000

    y_sig, y_ext = crystal_ball.pdf_ext(x, x_lo, x_hi, n_sig, *par_array)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-8)

    normalization = np.diff(
        scipy_crystal_ball.cdf(x[np.array([0, -1])], beta, m, mu, sigma)
    )
    y_norm = crystal_ball.pdf_norm(x, x[0], x[-1], *par_array)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_crystalball_cdf():
    x = np.arange(-10, 12)
    beta = 2
    m = 3
    mu = 1
    sigma = 2

    par_array = [mu, sigma, beta, m]

    y = crystal_ball.cdf(x, *par_array)
    y_direct = crystal_ball.get_cdf(x, *par_array)
    scipy_y = scipy_crystal_ball.cdf(x, beta, m, mu, sigma)

    assert isinstance(crystal_ball, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = crystal_ball.cdf_ext(x, n_sig, *par_array)
    assert np.allclose(y_ext, 20 * scipy_y, rtol=1e-8)

    normalization = np.diff(
        scipy_crystal_ball.cdf(x[np.array([0, -1])], beta, m, mu, sigma)
    )
    y_norm = crystal_ball.cdf_norm(x, x[0], x[-1], *par_array)

    assert np.allclose(y_norm, scipy_y / normalization, rtol=1e-8)


def test_required_args():
    names = crystal_ball.required_args()
    assert names[0] == "mu"
    assert names[1] == "sigma"
    assert names[2] == "beta"
    assert names[3] == "m"


def test_name():
    assert crystal_ball.name == "crystal_ball"
