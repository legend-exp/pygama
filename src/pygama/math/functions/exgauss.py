"""
Exponentially modified Gaussian distributions for pygama
"""

import sys
from math import erf, erfc

import numba as nb
import numpy as np

from pygama.math.functions.gauss import nb_gauss_pdf
from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.utils import numba_math_defaults as nb_defaults
from pygama.utils import numba_math_defaults_kwargs as nb_kwargs

limit = np.log(sys.float_info.max) / 10


@nb.njit(**nb_defaults(parallel=False))
def nb_gauss_tail_exact(
    x: float, mu: float, sigma: float, tau: float, tmp: float
) -> float:
    r"""
    Exact form of a normalized exponentially modified Gaussian PDF.
    It computes the following PDF:


    .. math::
        pdf(x, \tau,\mu,\sigma) = \frac{1}{2|\tau|}e^{\frac{x-\mu}{\tau}+\frac{\sigma^2}{2\tau^2}}\text{erfc}\left(\frac{\tau(\frac{x-\mu}{\sigma})+\sigma}{|\tau|\sqrt{2}}\right)


    Where :math:`tmp = \frac{x-\mu}{\tau}+\frac{\sigma^2}{2\tau^2}` is precomputed in :func:`nb_exgauss_pdf` to save computational time.


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
    tau
        The characteristic scale of the Gaussian tail
    tmp
        The scaled version of the exponential argument


    See Also
    --------
    :func:`nb_exgauss_pdf`
    """

    abstau = np.absolute(tau)
    if tmp < limit:
        tmp = tmp
    else:
        tmp = limit
    if sigma == 0 or abstau == 0:
        return x * 0
    z = (x - mu) / sigma
    tail_f = (
        (1 / (2 * abstau))
        * np.exp(tmp)
        * erfc((tau * z + sigma) / (np.sqrt(2) * abstau))
    )
    return tail_f


@nb.njit(**nb_defaults(parallel=False))
def nb_gauss_tail_approx(
    x: np.ndarray, mu: float, sigma: float, tau: float
) -> np.ndarray:
    r"""
    Approximate form of a normalized exponentially modified Gaussian PDF
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
    tau
        The characteristic scale of the Gaussian tail


    See Also
    --------
    :func:`nb_exgauss_pdf`
    """
    if sigma == 0:
        return x * 0
    elif (sigma + tau * (x - mu) / sigma) == 0:
        return x * 0
    den = 1 / (sigma + tau * (x - mu) / sigma)
    tail_f = sigma * nb_gauss_pdf(x, mu, sigma) * den * (1.0 - tau * tau * den * den)
    return tail_f


@nb.njit(**nb_kwargs)
def nb_exgauss_pdf(x: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
    r"""
    Normalized PDF of an exponentially modified Gaussian distribution. Its range of support is :math:`x\in(-\infty,\infty)`, :math:`\tau\in(-\infty,\infty)`
    Calls either :func:`nb_gauss_tail_exact` or :func:`nb_gauss_tail_approx` depending on which is computationally cheaper


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
    tau
        The characteristic scale of the Gaussian tail


    See Also
    --------
    :func:`nb_gauss_tail_exact`, :func:`nb_gauss_tail_approx`
    """

    x = np.asarray(x)
    tail_f = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        if tau == 0:
            tail_f[i] = np.nan
        else:
            tmp = ((x[i] - mu) / tau) + ((sigma**2) / (2 * tau**2))
            if tmp < limit:
                tail_f[i] = nb_gauss_tail_exact(x[i], mu, sigma, tau, tmp)
            else:
                tail_f[i] = nb_gauss_tail_approx(x[i], mu, sigma, tau)
    return tail_f


@nb.njit(**nb_kwargs)
def nb_exgauss_cdf(x: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
    r"""
    The CDF for a normalized exponentially modified Gaussian.  Its range of support is :math:`x\in(-\infty,\infty)`, :math:`\tau\in(-\infty,\infty)`
    It computes the following CDF:


    .. math::
        cdf(x,\tau,\mu,\sigma) = \tau\,pdf(x,\tau,\mu,\sigma)+ \frac{\tau}{2|\tau|}\text{erf}\left(\frac{\tau}{|\tau|\sqrt{2}}(\frac{x-\mu}{\sigma})\right) + \frac{1}{2}


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
    tau
        The characteristic scale of the Gaussian tail
    """
    abstau = np.abs(tau)

    cdf = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        if tau == 0:
            cdf[i] = np.nan
        else:
            cdf[i] = (tau / (2 * abstau)) * erf(
                (tau * (x[i] - mu)) / (np.sqrt(2) * sigma * abstau)
            )
            tmp = ((x[i] - mu) / tau) + ((sigma**2) / (2 * tau**2))
            if tmp < limit:
                cdf[i] += (
                    tau * nb_gauss_tail_exact(x[i], mu, sigma, tau, tmp) + 0.5
                )  # This is duplicated code from the pdf, but putting it in parallel makes it run faster!
            else:
                cdf[i] += tau * nb_gauss_tail_approx(x[i], mu, sigma, tau) + 0.5
    return cdf


@nb.njit(**nb_defaults(parallel=False))
def nb_exgauss_scaled_pdf(
    x: np.ndarray, area: float, mu: float, sigma: float, tau: float
) -> np.ndarray:
    r"""
    Scaled PDF of an exponentially modified Gaussian distribution
    Can be used as a component of other fit functions w/args mu,sigma,tau
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
    tau
        The characteristic scale of the Gaussian tail
    """

    return area * nb_exgauss_pdf(x, mu, sigma, tau)


@nb.njit(**nb_defaults(parallel=False))
def nb_exgauss_scaled_cdf(
    x: np.ndarray, area: float, mu: float, sigma: float, tau: float
) -> np.ndarray:
    r"""
    Scaled CDF of an exponentially modified Gaussian distribution
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
    tau
        The characteristic scale of the Gaussian tail
    """

    return area * nb_exgauss_cdf(x, mu, sigma, tau)


class ExgaussGen(PygamaContinuous):

    def __init__(self, *args, **kwargs):
        self.x_lo = -1 * np.inf
        self.x_hi = np.inf
        super().__init__(*args, **kwargs)

    def _pdf(self, x: np.ndarray, sigma: float, tau: float) -> np.ndarray:
        x.flags.writeable = True
        return nb_exgauss_pdf(
            x, 0, 1, tau[0] / sigma[0]
        )  # the scipy parameter k = tau/sigma

    def _cdf(self, x: np.ndarray, sigma: float, tau: float) -> np.ndarray:
        x.flags.writeable = True
        return nb_exgauss_cdf(x, 0, 1, tau[0] / sigma[0])

    def get_pdf(self, x: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
        return nb_exgauss_pdf(x, mu, sigma, tau)

    def get_cdf(self, x: np.ndarray, mu: float, sigma: float, tau: float) -> np.ndarray:
        return nb_exgauss_cdf(x, mu, sigma, tau)

    def pdf_norm(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        tau: float,
    ) -> np.ndarray:
        return self._pdf_norm(x, x_lo, x_hi, mu, sigma, tau)

    def cdf_norm(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        tau: float,
    ) -> np.ndarray:
        return self._cdf_norm(x, x_lo, x_hi, mu, sigma, tau)

    def pdf_ext(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        area: float,
        mu: float,
        sigma: float,
        tau: float,
    ) -> np.ndarray:
        return np.diff(
            nb_exgauss_scaled_cdf(np.array([x_lo, x_hi]), area, mu, sigma, tau)
        )[0], nb_exgauss_scaled_pdf(x, area, mu, sigma, tau)

    def cdf_ext(
        self, x: np.ndarray, area: float, mu: float, sigma: float, tau: float
    ) -> np.ndarray:
        return nb_exgauss_scaled_cdf(x, area, mu, sigma, tau)

    def required_args(self) -> tuple[str, str, str]:
        return "mu", "sigma", "tau"


exgauss = ExgaussGen(name="exgauss")
