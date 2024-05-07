"""
Exponential distributions for pygama
"""

import numba as nb
import numpy as np

from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.utils import numba_math_defaults as nb_defaults
from pygama.utils import numba_math_defaults_kwargs as nb_kwargs


@nb.njit(**nb_kwargs)
def nb_exponential_pdf(
    x: np.ndarray, mu: float, sigma: float, lamb: float
) -> np.ndarray:
    r"""
    Normalised exponential probability density distribution, w/ args: mu, sigma, lamb. Its range of support is :math:`x\in[0,\infty), \lambda>0`.
    It computes:


    .. math::
        pdf(x, \lambda, \mu, \sigma) = \begin{cases} \lambda e^{-\lambda\frac{x-\mu}{\sigma}} \quad , \frac{x-\mu}{\sigma}\geq 0 \\ 0 \quad , \frac{x-\mu}{\sigma}<0 \end{cases}


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        The input data
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    lamb
        The rate
    """

    y = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        y[i] = (x[i] - mu) / sigma
        if y[i] < 0:
            y[i] = 0
        else:
            y[i] = (lamb * np.exp(-1 * lamb * y[i])) / sigma
    return y


@nb.njit(**nb_kwargs)
def nb_exponential_cdf(
    x: np.ndarray, mu: float, sigma: float, lamb: float
) -> np.ndarray:
    r"""
    Normalised exponential cumulative distribution, w/ args:  mu, sigma, lamb. Its range of support is :math:`x\in[0,\infty), \lambda>0`.
    It computes:


    .. math::
        cdf(x, \lambda, \mu, \sigma) = \begin{cases}  1-e^{-\lambda\frac{x-\mu}{\sigma}} \quad , \frac{x-\mu}{\sigma} > 0 \\ 0 \quad , \frac{x-\mu}{\sigma}\leq 0 \end{cases}


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        The input data
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    lamb
        The rate
    """

    y = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        y[i] = (x[i] - mu) / sigma
        if y[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1 - np.exp(-1 * lamb * y[i])
    return y


@nb.njit(**nb_defaults(parallel=False))
def nb_exponential_scaled_pdf(
    x: np.ndarray, area: float, mu: float, sigma: float, lamb: float
) -> np.ndarray:
    r"""
    Scaled exponential probability distribution, w/ args: area, mu, sigma, lambd.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        Input data
    area
        The number of counts in the signal
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    lamb
        The rate
    """

    return area * nb_exponential_pdf(x, mu, sigma, lamb)


@nb.njit(**nb_defaults(parallel=False))
def nb_exponential_scaled_cdf(
    x: np.ndarray, area: float, mu: float, sigma: float, lamb: float
) -> np.ndarray:
    r"""
    Exponential cdf scaled by the area, used for extended binned fits
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        Input data
    area
        The number of counts in the signal
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    lamb
        The rate
    """

    return area * nb_exponential_cdf(x, mu, sigma, lamb)


class ExponentialGen(PygamaContinuous):

    def __init__(self, *args, **kwargs):
        self.x_lo = 0
        self.x_hi = np.inf
        super().__init__(*args, **kwargs)

    def _pdf(self, x: np.ndarray, mu: float, sigma: float, lamb: float) -> np.ndarray:
        x.flags.writeable = True
        return nb_exponential_pdf(x, mu[0], sigma[0], lamb[0])

    def _cdf(self, x: np.ndarray, mu: float, sigma: float, lamb: float) -> np.ndarray:
        x.flags.writeable = True
        return nb_exponential_cdf(x, mu[0], sigma[0], lamb[0])

    def get_pdf(
        self, x: np.ndarray, mu: float, sigma: float, lamb: float
    ) -> np.ndarray:
        return nb_exponential_pdf(x, mu, sigma, lamb)

    def get_cdf(
        self, x: np.ndarray, mu: float, sigma: float, lamb: float
    ) -> np.ndarray:
        return nb_exponential_cdf(x, mu, sigma, lamb)

    # needed so that we can hack iminuit's introspection to function parameter names... unless
    def pdf_norm(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        lamb: float,
    ) -> np.ndarray:
        return self._pdf_norm(x, x_lo, x_hi, mu, sigma, lamb)

    def cdf_norm(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        lamb: float,
    ) -> np.ndarray:
        return self._cdf_norm(x, x_lo, x_hi, mu, sigma, lamb)

    def pdf_ext(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        area: float,
        mu: float,
        sigma: float,
        lamb: float,
    ) -> np.ndarray:
        return np.diff(
            nb_exponential_scaled_cdf(np.array([x_lo, x_hi]), area, mu, sigma, lamb)
        )[0], nb_exponential_scaled_pdf(x, area, mu, sigma, lamb)

    def cdf_ext(
        self, x: np.ndarray, area: float, mu: float, sigma: float, lamb: float
    ) -> np.ndarray:
        return nb_exponential_scaled_cdf(x, area, mu, sigma, lamb)

    def required_args(self) -> tuple[str, str, str]:
        return "mu", "sigma", "lambda"


exponential = ExponentialGen(a=0.0, name="exponential")
