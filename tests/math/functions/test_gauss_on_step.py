import numpy as np
from scipy.stats import norm

from pygama.math.functions.gauss_on_step import gauss_on_step
from pygama.math.functions.sum_dists import SumDists


def test_gauss_on_step_pdf():
    x = np.arange(-10, 10)
    mu = 0.4
    sigma = 1.1
    hstep = 0.75
    x_lo = np.amin(x)
    x_hi = np.amax(x)
    n_sig = 2
    n_bkg = 4

    pars = np.array([x_lo, x_hi, n_sig, mu, sigma, n_bkg, hstep], dtype=float)

    y_direct = gauss_on_step.get_pdf(x, *pars)

    # compute the unnormalized step function
    scipy_step = 1 + hstep * (
        norm.cdf(x, mu, sigma) * 2 - 1
    )  # pdf = (1+erf(x/np.sqrt(2))); erf(x) = 2 cdf_norm(x*sqrt(2))-1

    # compute the normalization for the step function
    maximum = (np.amax(x) - mu) / sigma
    minimum = (np.amin(x) - mu) / sigma
    normalization = sigma * (
        (1 - hstep) * (maximum - minimum)
        + 2
        * hstep
        * (
            maximum * norm.cdf(maximum)
            + np.exp(-1 * maximum**2 / 2) / np.sqrt(2 * np.pi)
            - minimum * norm.cdf(minimum)
            - np.exp(-1 * minimum**2 / 2) / np.sqrt(2 * np.pi)
        )
    )

    # compute the normalized step function
    scipy_y_step = scipy_step / normalization

    scipy_y_gauss = norm.pdf(x, mu, sigma)
    scipy_y = n_sig * scipy_y_gauss + n_bkg * scipy_y_step

    assert isinstance(gauss_on_step, SumDists)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_sig, y_ext = gauss_on_step.pdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig + n_bkg, rtol=1e-8)

    y_sig = gauss_on_step.pdf_norm(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)


def test_gauss_on_step_cdf():
    x = np.arange(-10, 10)
    mu = 0.56
    sigma = 1.009
    hstep = 0.753
    x_lo = np.amin(x)
    x_hi = np.amax(x)
    n_sig = 2
    n_bkg = 4

    pars = np.array([x_lo, x_hi, n_sig, mu, sigma, n_bkg, hstep], dtype=float)

    y_direct = gauss_on_step.get_cdf(x, *pars)

    # Compute the normalization of the pdf
    maximum = (np.amax(x) - mu) / sigma
    minimum = (np.amin(x) - mu) / sigma
    pdf_normalization = sigma * (
        (1 - hstep) * (maximum - minimum)
        + 2
        * hstep
        * (
            maximum * norm.cdf(maximum)
            + np.exp(-1 * maximum**2 / 2) / np.sqrt(2 * np.pi)
            - minimum * norm.cdf(minimum)
            - np.exp(-1 * minimum**2 / 2) / np.sqrt(2 * np.pi)
        )
    )

    z = (x - mu) / sigma
    # Compute the unnormalized cdf
    unnormalized_cdf = sigma * (
        (1 - hstep) * (z - minimum)
        + 2
        * hstep
        * (
            z * norm.cdf(z)
            + np.exp(-1 * z**2 / 2) / np.sqrt(2 * np.pi)
            - minimum * norm.cdf(minimum)
            - np.exp(-1 * minimum**2 / 2) / np.sqrt(2 * np.pi)
        )
    )

    # Compute the cdf
    scipy_y_step = unnormalized_cdf / pdf_normalization

    scipy_y_gauss = norm.cdf(x, mu, sigma)
    # for the cdf, we actually do not want to normalize by the sum of the areas, this is because iminuit takes a total unnormalized cdf for extended binned fits
    scipy_y = n_sig * scipy_y_gauss + n_bkg * scipy_y_step

    assert isinstance(gauss_on_step, SumDists)

    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    y_ext = gauss_on_step.cdf_ext(x, *pars)
    assert np.allclose(y_ext, scipy_y, rtol=1e-8)

    y_norm = gauss_on_step.cdf_norm(x, *pars)
    scipy_y_norm = (1 / (n_sig + n_bkg)) * (
        n_sig * scipy_y_gauss + n_bkg * scipy_y_step
    )

    assert np.allclose(y_norm, scipy_y_norm, rtol=1e-8)


def test_required_args():
    names = gauss_on_step.required_args()
    assert names[0] == "x_lo"
    assert names[1] == "x_hi"
    assert names[2] == "n_sig"
    assert names[3] == "mu"
    assert names[4] == "sigma"
    assert names[5] == "n_bkg"
    assert names[6] == "hstep"


def test_name():
    assert gauss_on_step.name == "gauss_on_step"
