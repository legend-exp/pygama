import numpy as np
from scipy.stats import expon, norm

from pygama.math.functions.gauss_on_exponential import gauss_on_exponential
from pygama.math.functions.sum_dists import SumDists


def test_gauss_on_exponential_pdf():
    x = np.arange(-10, 10)
    n_sig = 200
    mu = 1
    sigma = 1
    n_bkg = 100
    lamb = 3
    mu_exp = 2
    sigma_exp = 0.2

    pars = np.array([n_sig, mu, sigma, n_bkg, lamb, mu_exp, sigma_exp], dtype=float)

    assert isinstance(gauss_on_exponential, SumDists)

    y_direct = gauss_on_exponential.get_pdf(x, *pars)
    scipy_exponential = n_bkg * expon.pdf(x, mu_exp, sigma_exp / lamb)
    scipy_gauss = n_sig * norm.pdf(x, mu, sigma)

    scipy_y = scipy_exponential + scipy_gauss

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    x_lo = -100
    x_hi = 100

    pars = np.array(
        [x_lo, x_hi, n_sig, mu, sigma, n_bkg, lamb, mu_exp, sigma_exp], dtype=float
    )

    y_sig, y_ext = gauss_on_exponential.pdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig + n_bkg, rtol=1e-8)


def test_gauss_on_exponential_cdf():
    x = np.arange(-10, 10)
    n_sig = 200
    mu = 1
    sigma = 1
    n_bkg = 100
    lamb = 3
    mu_exp = 2
    sigma_exp = 0.2

    pars = np.array([n_sig, mu, sigma, n_bkg, lamb, mu_exp, sigma_exp], dtype=float)

    assert isinstance(gauss_on_exponential, SumDists)

    y_direct = gauss_on_exponential.get_cdf(x, *pars)

    scipy_exponential = n_bkg * expon.cdf(x, mu_exp, sigma_exp / lamb)
    scipy_gauss = n_sig * norm.cdf(x, mu, sigma)

    scipy_y = scipy_exponential + scipy_gauss

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_ext = gauss_on_exponential.cdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)


def test_required_args():
    names = gauss_on_exponential.required_args()
    assert names[0] == "n_sig"
    assert names[1] == "mu"
    assert names[2] == "sigma"
    assert names[3] == "n_bkg"
    assert names[4] == "lambd"
    assert names[5] == "mu_exp"
    assert names[6] == "sigma_exp"


def test_name():
    assert gauss_on_exponential.name == "gauss_on_exponential"
