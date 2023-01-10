import numpy as np
from scipy.stats import norm

from pygama.math.functions.pygama_continuous import pygama_continuous
from pygama.math.functions.step import step


def test_step_pdf():

    x = np.arange(-10, 10)
    mu = 2.3
    sigma = 1.1
    hstep = 0.65
    lower_range = np.inf
    upper_range = np.inf

    y = step.pdf(x, hstep, lower_range, upper_range, mu, sigma)
    y_direct = step.get_pdf(x, hstep, lower_range, upper_range, mu, sigma)

    # compute the unnormalized step function
    scipy_step = 1 + hstep * (
        norm.cdf(x, mu, sigma) * 2 - 1
    )  # pdf = (1+erf(x/np.sqrt(2))); erf(x) = 2 norm_cdf(x*sqrt(2))-1

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
    scipy_y = scipy_step / normalization

    assert isinstance(step, pygama_continuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    x_lo = -10
    x_hi = 10
    y_sig, y_ext = step.pdf_ext(
        x, n_sig, x_lo, x_hi, hstep, lower_range, upper_range, mu, sigma
    )
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-8)

    y_norm = step.norm_pdf(x, x[0], x[-1], hstep, mu, sigma)

    assert np.allclose(y_norm, scipy_y, rtol=1e-8)


def test_step_cdf():

    x = np.arange(-10, 10)
    mu = 1.1
    sigma = 2.3
    hstep = 4.5
    lower_range = np.inf
    upper_range = np.inf

    y = step.cdf(x, hstep, lower_range, upper_range, mu, sigma)
    y_direct = step.get_cdf(x, hstep, lower_range, upper_range, mu, sigma)

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
    scipy_y = unnormalized_cdf / pdf_normalization

    assert isinstance(step, pygama_continuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = step.cdf_ext(x, n_sig, hstep, lower_range, upper_range, mu, sigma)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)

    y_norm = step.norm_cdf(x, x[0], x[-1], hstep, mu, sigma)

    assert np.allclose(y_norm, scipy_y, rtol=1e-8)
