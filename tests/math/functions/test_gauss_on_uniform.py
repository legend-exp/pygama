import numpy as np
from scipy.stats import norm, uniform

from pygama.math.functions.gauss_on_uniform import gauss_on_uniform
from pygama.math.functions.sum_dists import SumDists


def test_gauss_on_uniform_pdf():
    x = np.arange(-10, 10)
    mu = 1
    sigma = 2
    n_sig = 10
    n_bkg = 20

    pars = np.array([np.amin(x), np.amax(x), n_sig, mu, sigma, n_bkg], dtype=float)

    assert isinstance(gauss_on_uniform, SumDists)

    y_direct = gauss_on_uniform.get_pdf(x, *pars)

    scipy_gauss = n_sig * norm.pdf(x, mu, sigma)
    scipy_uniform = n_bkg * uniform.pdf(x, np.amin(x), np.amax(x))

    scipy_y = scipy_gauss + scipy_uniform

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_sig, y_ext = gauss_on_uniform.pdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig + n_bkg, rtol=1e-3)


def test_gauss_on_uniform_cdf():
    x = np.arange(-10, 10)
    mu = 1
    sigma = 2
    n_sig = 10
    n_bkg = 20

    pars = np.array([np.amin(x), np.amax(x), n_sig, mu, sigma, n_bkg], dtype=float)

    assert isinstance(gauss_on_uniform, SumDists)

    y_direct = gauss_on_uniform.get_cdf(x, *pars)

    scipy_gauss = n_sig * norm.cdf(x, mu, sigma)
    scipy_uniform = n_bkg * uniform.cdf(x, np.amin(x), np.amax(x))

    scipy_y = scipy_gauss + scipy_uniform

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_ext = gauss_on_uniform.cdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)


def test_required_args():
    names = gauss_on_uniform.required_args()
    assert names[0] == "x_lo"
    assert names[1] == "x_hi"
    assert names[2] == "n_sig"
    assert names[3] == "mu"
    assert names[4] == "sigma"
    assert names[5] == "n_bkg"


def test_name():
    assert gauss_on_uniform.name == "gauss_on_uniform"
