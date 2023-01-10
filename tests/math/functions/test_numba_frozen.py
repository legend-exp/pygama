import numpy as np
from scipy.stats import norm

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.pygama_continuous import numba_frozen


def test_numba_frozen():
    x = np.arange(-10, 10)
    mu = 2
    sigma = 3
    gauss = gaussian(mu, sigma)
    assert isinstance(gauss, numba_frozen)

    numba_pdf = gauss.pdf(x)
    scipy_pdf = norm.pdf(x, mu, sigma)
    assert np.allclose(numba_pdf, scipy_pdf)

    numba_get_pdf = gauss.get_pdf(x)
    assert np.allclose(numba_get_pdf, scipy_pdf)

    numba_cdf = gauss.cdf(x)
    scipy_cdf = norm.cdf(x, mu, sigma)
    assert np.allclose(numba_cdf, scipy_cdf)

    numba_get_cdf = gauss.get_cdf(x)
    assert np.allclose(numba_get_cdf, scipy_cdf)

    area = 10
    x_lo = -20
    x_hi = 20
    numba_extended_pdf_area, numba_extended_pdf = gauss.pdf_ext(x, area, x_lo, x_hi)
    assert np.allclose(numba_extended_pdf_area, area, rtol=1e-3)
    assert np.allclose(numba_extended_pdf, area * scipy_pdf)

    numba_extended_cdf = gauss.cdf_ext(x, area)
    assert np.allclose(numba_extended_cdf, area * scipy_cdf)
