import numpy as np
from scipy.stats import norm

from pygama.math.functions.sum_dists import SumDists
from pygama.math.functions.triple_gauss_on_double_step import (
    triple_gauss_on_double_step,
)


def test_triple_gauss_on_double_step_pdf():
    x = np.arange(-50, 50)
    mu_1 = 0
    sigma_1 = 1
    hstep_1 = 1

    mu_2 = mu_1
    sigma_2 = sigma_1
    hstep_2 = hstep_1

    mu_3 = 5
    sigma_3 = 10

    x_lo = np.amin(x)
    x_hi = np.amax(x)
    n_sig1 = 2
    n_sig2 = 10
    n_sig3 = 4
    n_bkg1 = 4
    n_bkg2 = 3

    pars = np.array(
        [
            x_lo,
            x_hi,
            n_sig1,
            mu_1,
            sigma_1,
            n_sig2,
            mu_2,
            sigma_2,
            n_sig3,
            mu_3,
            sigma_3,
            n_bkg1,
            hstep_1,
            n_bkg2,
            hstep_2,
        ],
        dtype=float,
    )

    n_sig_array = [n_sig1, n_sig2, n_sig3, n_bkg1, n_bkg2]

    # compute the unnormalized step function
    scipy_y_step = n_bkg1 * (
        1 + hstep_1 * (norm.cdf(x, mu_1, sigma_1) * 2 - 1)
    ) + n_bkg2 * (
        1 + hstep_2 * (norm.cdf(x, mu_2, sigma_2) * 2 - 1)
    )  # pdf = (1+erf(x/np.sqrt(2)))/18; erf(x) = 2 cdf_norm(x*sqrt(2))-1

    maximum = (np.amax(x) - mu_1) / sigma_1
    minimum = (np.amin(x) - mu_1) / sigma_1
    normalization = sigma_1 * (
        (1 - hstep_1) * (maximum - minimum)
        + 2
        * hstep_1
        * (
            maximum * norm.cdf(maximum)
            + np.exp(-1 * maximum**2 / 2) / np.sqrt(2 * np.pi)
            - minimum * norm.cdf(minimum)
            - np.exp(-1 * minimum**2 / 2) / np.sqrt(2 * np.pi)
        )
    )
    scipy_y_step = scipy_y_step / normalization

    y_direct = triple_gauss_on_double_step.get_pdf(x, *pars)

    scipy_y_gauss = (
        n_sig1 * norm.pdf(x, mu_1, sigma_1)
        + n_sig2 * norm.pdf(x, mu_2, sigma_2)
        + n_sig3 * norm.pdf(x, mu_3, sigma_3)
    )

    scipy_y = scipy_y_step + scipy_y_gauss

    assert isinstance(triple_gauss_on_double_step, SumDists)

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_sig, y_ext = triple_gauss_on_double_step.pdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
    assert np.allclose(
        y_sig, np.sum(n_sig_array), rtol=1e-4
    )  # We aren't using trun_norm, so this shouldn't be quite equal to the sum of the areas, but quite close


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

    x_lo = np.amin(x)
    x_hi = np.amax(x)
    n_sig1 = 2
    n_sig2 = 10
    n_sig3 = 4
    n_bkg1 = 4
    n_bkg2 = 3

    pars = np.array(
        [
            x_lo,
            x_hi,
            n_sig1,
            mu_1,
            sigma_1,
            n_sig2,
            mu_2,
            sigma_2,
            n_sig3,
            mu_3,
            sigma_3,
            n_bkg1,
            hstep_1,
            n_bkg2,
            hstep_2,
        ],
        dtype=float,
    )

    y_direct = triple_gauss_on_double_step.get_cdf(x, *pars)
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

    assert isinstance(triple_gauss_on_double_step, SumDists)

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_ext = triple_gauss_on_double_step.cdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)


def test_required_args():
    names = triple_gauss_on_double_step.required_args()
    assert names[0] == "x_lo"
    assert names[1] == "x_hi"
    assert names[2] == "n_sig1"
    assert names[3] == "mu1"
    assert names[4] == "sigma1"
    assert names[5] == "n_sig2"
    assert names[6] == "mu2"
    assert names[7] == "sigma2"
    assert names[8] == "n_sig3"
    assert names[9] == "mu3"
    assert names[10] == "sigma3"
    assert names[11] == "n_bkg1"
    assert names[12] == "hstep1"
    assert names[13] == "n_bkg2"
    assert names[14] == "hstep2"


def test_name():
    assert triple_gauss_on_double_step.name == "triple_gauss_on_double_step"
