import numpy as np
from scipy.stats import norm

from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.math.functions.step import step


def test_step_pdf():
    x = np.arange(-10, 10)
    mu = 2.3
    sigma = 1.1
    hstep = 0.65
    x_lo = np.amin(x)
    x_hi = np.amax(x)

    param_array = [x_lo, x_hi, mu, sigma, hstep]

    y = step.pdf(x, *param_array)
    y_direct = step.get_pdf(x, *param_array)

    # compute the unnormalized step function
    scipy_step = 1 + hstep * (
        norm.cdf(x, mu, sigma) * 2 - 1
    )  # pdf = (1+erf(x/np.sqrt(2))); erf(x) = 2 cdf_norm(x*sqrt(2))-1

    # compute the normalization for the step function
    maximum = (x_hi - mu) / sigma
    minimum = (x_lo - mu) / sigma
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

    assert isinstance(step, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    x_lo = -10
    x_hi = 9

    y_sig, y_ext = step.pdf_ext(x, x_lo, x_hi, n_sig, mu, sigma, hstep)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)
    assert np.allclose(y_sig, n_sig, rtol=1e-8)

    y_norm = step.pdf_norm(x, x_lo, x_hi, mu, sigma, hstep)

    assert np.allclose(y_norm, scipy_y, rtol=1e-8)


def test_step_cdf():
    x = np.arange(-10, 10)
    mu = 1.1
    sigma = 2.3
    hstep = 4.5
    x_lo = np.amin(x)
    x_hi = np.amax(x)

    param_array = [x_lo, x_hi, mu, sigma, hstep]

    y = step.cdf(x, *param_array)
    y_direct = step.get_cdf(x, *param_array)

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

    assert isinstance(step, PygamaContinuous)
    assert np.allclose(y, scipy_y, rtol=1e-8)
    assert np.allclose(y_direct, scipy_y, rtol=1e-8)

    n_sig = 20
    y_ext = step.cdf_ext(x, x_lo, x_hi, n_sig, mu, sigma, hstep)
    assert np.allclose(y_ext, n_sig * scipy_y, rtol=1e-8)

    y_norm = step.cdf_norm(x, x[0], x[-1], mu, sigma, hstep)

    assert np.allclose(y_norm, scipy_y, rtol=1e-8)


def test_required_args():
    names = step.required_args()
    assert names[0] == "x_lo"
    assert names[1] == "x_hi"
    assert names[2] == "mu"
    assert names[3] == "sigma"
    assert names[4] == "hstep"


def test_name():
    assert step.name == "step"
