import numpy as np
from scipy.stats import norm

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.pygama_continuous import NumbaFrozen


def test_numba_frozen():
    x = np.arange(-10, 10)
    mu = 2
    sigma = 3
    gauss = gaussian(mu, sigma)
    assert isinstance(gauss, NumbaFrozen)

    numba_pdf = gauss.pdf(x)
    scipy_pdf = norm.pdf(x, mu, sigma)
    assert np.allclose(numba_pdf, scipy_pdf)

    numba_get_pdf = gauss.get_pdf(x)
    assert np.allclose(numba_get_pdf, scipy_pdf)

    numba_pdf_norm = gauss.pdf_norm(x, x[0], x[-1])
    scipy_pdf_norm = scipy_pdf / (np.diff(norm.cdf([x[0], x[-1]], mu, sigma)))
    assert np.allclose(numba_pdf_norm, scipy_pdf_norm)

    numba_cdf = gauss.cdf(x)
    scipy_cdf = norm.cdf(x, mu, sigma)
    assert np.allclose(numba_cdf, scipy_cdf)

    numba_get_cdf = gauss.get_cdf(x)
    assert np.allclose(numba_get_cdf, scipy_cdf)

    numba_cdf_norm = gauss.cdf_norm(x, x[0], x[-1])
    scipy_cdf_norm = scipy_cdf / (np.diff(norm.cdf([x[0], x[-1]], mu, sigma)))
    assert np.allclose(numba_cdf_norm, scipy_cdf_norm)

    area = 10
    x_lo = -20
    x_hi = 20

    numba_extended_pdf_area, numba_extended_pdf = gauss.pdf_ext(x, x_lo, x_hi, area)
    assert np.allclose(numba_extended_pdf_area, area, rtol=1e-3)
    assert np.allclose(numba_extended_pdf, area * scipy_pdf)

    numba_extended_cdf = gauss.cdf_ext(x, area)
    assert np.allclose(numba_extended_cdf, area * scipy_cdf)
