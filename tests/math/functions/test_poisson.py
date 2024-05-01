import numpy as np
from scipy.stats import poisson as scipy_poisson
from scipy.stats import rv_discrete

from pygama.math.functions.poisson import poisson


def test_poisson_pdf():
    x = np.arange(1, 12)
    mu = 1
    lamb = 2

    y = poisson.pmf(x, mu, lamb)
    y_direct = poisson.get_pmf(x, mu, lamb)
    scipy_y = scipy_poisson.pmf(x, lamb, mu)

    assert isinstance(poisson, rv_discrete)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    x_lo = 0
    x_hi = 12

    y_sig, y_ext = poisson.pmf_ext(x, x_lo, x_hi, n_sig, mu, lamb)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-3)


def test_poisson_cdf():
    x = np.arange(0, 12)
    mu = 1
    lamb = 2

    y = poisson.cdf(x, mu, lamb)
    y_direct = poisson.get_cdf(x, mu, lamb)
    scipy_y = scipy_poisson.cdf(x, lamb, mu)

    assert isinstance(poisson, rv_discrete)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = poisson.cdf_ext(x, n_sig, mu, lamb)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)


def test_required_args():
    names = poisson.required_args()
    assert names[0] == "mu"
    assert names[1] == "lamb"


def test_name():
    assert poisson.name == "poisson"
