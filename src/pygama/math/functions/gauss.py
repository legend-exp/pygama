"""
Gaussian distributions for pygama
"""

import numba as nb
import numpy as np

from pygama.math.functions.error_function import nb_erf
from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.utils import numba_math_defaults as nb_defaults


@nb.njit(**nb_defaults(parallel=False))
def nb_gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Gaussian, unnormalised for use in building PDFs
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        Input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian

    .. note::
        TODO:: remove this in favor of using nb_gauss_pdf with a different normalization
    """

    if sigma == 0:
        invs = np.inf
    else:
        invs = 1.0 / sigma
    z = (x - mu) * invs
    return np.exp(-0.5 * z**2)


@nb.njit(**nb_defaults(parallel=False))
def nb_gauss_amp(x: np.ndarray, mu: float, sigma: float, a: float) -> np.ndarray:
    """
    Gaussian with height as a parameter for FWHM etc.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        Input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    a
        The amplitude of the Gaussian

    .. note::
        TODO:: potentially remove this, redundant with ``nb_gauss_scaled_pdf``
    """

    if sigma == 0:
        invs = np.inf
    else:
        invs = 1.0 / sigma
    z = (x - mu) * invs
    return a * np.exp(-0.5 * z**2)


@nb.njit(**nb_defaults(parallel=False))
def nb_gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    Normalised Gaussian PDF, w/ args: mu, sigma. The support is :math:`(-\infty, \infty)`

    .. math::
        pdf(x, \mu, \sigma) = \frac{1}{\sqrt{2\pi}}e^{(\frac{x-\mu}{\sigma}^2)/2}

    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    """

    if sigma == 0:
        invs = np.inf
    else:
        invs = 1.0 / sigma
    z = (x - mu) * invs
    invnorm = invs / np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z**2) * invnorm


@nb.njit(**nb_defaults(parallel=False))
def nb_gauss_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    Gaussian CDF, w/ args: mu, sigma. The support is :math:`(-\infty, \infty)`

    .. math::
        cdf(x, \mu,\sigma) =  \frac{1}{2}\left[1+\text{erf}(\frac{x-\mu}{\sigma\sqrt{2}})\right]

    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    """

    if sigma == 0:
        invs = np.inf
    else:
        invs = 1.0 / sigma

    return 1 / 2 * (1 + nb_erf(invs * (x - mu) / (np.sqrt(2))))


@nb.njit(**nb_defaults(parallel=False))
def nb_gauss_scaled_pdf(
    x: np.ndarray, area: float, mu: float, sigma: float
) -> np.ndarray:
    """
    Gaussian with height as a parameter for fwhm etc.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        Input data
    area
        The number of counts in the signal
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    """

    return area * nb_gauss_pdf(x, mu, sigma)


@nb.njit(**nb_defaults(parallel=False))
def nb_gauss_scaled_cdf(
    x: np.ndarray, area: float, mu: float, sigma: float
) -> np.ndarray:
    """
    Gaussian CDF scaled by the number of signal counts for extended binned fits
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        Input data
    area
        The number of counts in the signal
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    """

    return area * nb_gauss_cdf(x, mu, sigma)


class GaussianGen(PygamaContinuous):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_lo = -1 * np.inf
        self.x_hi = np.inf

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        x.flags.writeable = True
        return nb_gauss_pdf(x, 0, 1)

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        return nb_gauss_cdf(x, 0, 1)

    def get_pdf(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return nb_gauss_pdf(x, mu, sigma)

    def get_cdf(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return nb_gauss_cdf(x, mu, sigma)

    def pdf_norm(
        self, x: np.ndarray, x_lo: float, x_hi: float, mu: float, sigma: float
    ) -> np.ndarray:
        return self._pdf_norm(x, x_lo, x_hi, mu, sigma)

    def cdf_norm(
        self, x: np.ndarray, x_lo: float, x_hi: float, mu: float, sigma: float
    ) -> np.ndarray:
        return self._cdf_norm(x, x_lo, x_hi, mu, sigma)

    def pdf_ext(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        area: float,
        mu: float,
        sigma: float,
    ) -> np.ndarray:
        return np.diff(nb_gauss_scaled_cdf(np.array([x_lo, x_hi]), area, mu, sigma))[
            0
        ], nb_gauss_scaled_pdf(x, area, mu, sigma)

    def cdf_ext(
        self, x: np.ndarray, area: float, mu: float, sigma: float
    ) -> np.ndarray:
        return nb_gauss_scaled_cdf(x, area, mu, sigma)

    def required_args(self) -> tuple[str, str]:
        return "mu", "sigma"


gaussian = GaussianGen(name="gaussian")
