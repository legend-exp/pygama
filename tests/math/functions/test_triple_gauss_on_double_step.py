import numpy as np
from scipy.stats import norm

from pygama.math.functions.sum_dists import sum_dists
from pygama.math.functions.triple_gauss_on_double_step import (
    triple_gauss_on_double_step,
)


def test_triple_gauss_on_double_step_pdf():

    x = np.arange(-10, 10)
    mu_1 = 0
    sigma_1 = 1
    hstep_1 = 1

    mu_2 = 0
    sigma_2 = 1
    hstep_2 = 1

    mu_3 = 5
    sigma_3 = 10

    lower_range = np.inf
    upper_range = np.inf
    n_sig1 = 2
    n_sig2 = 10
    n_sig3 = 4
    n_bkg1 = 4
    n_bkg2 = 3

    pars = np.array(
        [
            mu_1,
            sigma_1,
            n_sig1,
            mu_2,
            sigma_2,
            n_sig2,
            mu_3,
            sigma_3,
            n_sig3,
            hstep_1,
            lower_range,
            upper_range,
            n_bkg1,
            hstep_2,
            n_bkg2,
        ],
        dtype=float,
    )

    n_sig_array = [n_sig1, n_sig2, n_sig3, n_bkg1, n_bkg2]

    y_direct = triple_gauss_on_double_step.get_pdf(x, pars)
    scipy_y_step = (
        n_bkg1 * norm.cdf(x, mu_1, sigma_1) * 2 / 18
        + n_bkg2 * norm.cdf(x, mu_2, sigma_2) * 2 / 18
    )  # pdf = (1+erf(x/np.sqrt(2)))/18; erf(x) = 2 norm_cdf(x*sqrt(2))-1
    scipy_y_gauss = (
        n_sig1 * norm.pdf(x, mu_1, sigma_1)
        + n_sig2 * norm.pdf(x, mu_2, sigma_2)
        + n_sig3 * norm.pdf(x, mu_3, sigma_3)
    )

    scipy_y = scipy_y_step + scipy_y_gauss

    assert isinstance(triple_gauss_on_double_step, sum_dists)

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    x_lo = -100
    x_hi = 100
    pars = np.insert(pars, 0, [x_lo, x_hi])
    y_sig, y_ext = triple_gauss_on_double_step.pdf_ext(x, pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, np.sum(n_sig_array), rtol=1e-8)


def test_triple_gauss_on_double_step_cdf():

    x = np.arange(-10, 10)
    mu_1 = 0
    sigma_1 = 1
    hstep_1 = 1

    mu_2 = 0
    sigma_2 = 1
    hstep_2 = 1

    mu_3 = 5
    sigma_3 = 10

    lower_range = np.inf
    upper_range = np.inf
    n_sig1 = 2
    n_sig2 = 10
    n_sig3 = 4
    n_bkg1 = 4
    n_bkg2 = 3

    pars = np.array(
        [
            mu_1,
            sigma_1,
            n_sig1,
            mu_2,
            sigma_2,
            n_sig2,
            mu_3,
            sigma_3,
            n_sig3,
            hstep_1,
            lower_range,
            upper_range,
            n_bkg1,
            hstep_2,
            n_bkg2,
        ],
        dtype=float,
    )

    y_direct = triple_gauss_on_double_step.get_cdf(x, pars)
    scipy_y_step = n_bkg1 * (
        (x * (2 * norm.cdf(x, mu_1, sigma_1) - 1) / np.sqrt(2))
        + np.exp(-1 * x**2 / 2) / np.sqrt(np.pi)
        + x / np.sqrt(2)
    ) * 2 / 18 / np.sqrt(2) + n_bkg2 * (
        (x * (2 * norm.cdf(x, mu_2, sigma_2) - 1) / np.sqrt(2))
        + np.exp(-1 * x**2 / 2) / np.sqrt(np.pi)
        + x / np.sqrt(2)
    ) * 2 / 18 / np.sqrt(
        2
    )
    scipy_y_gauss = (
        n_sig1 * norm.cdf(x, mu_1, sigma_1)
        + n_sig2 * norm.cdf(x, mu_2, sigma_2)
        + n_sig3 * norm.cdf(x, mu_3, sigma_3)
    )
    scipy_y = scipy_y_step + scipy_y_gauss

    assert isinstance(triple_gauss_on_double_step, sum_dists)

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_ext = triple_gauss_on_double_step.cdf_ext(x, pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
