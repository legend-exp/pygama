"""
Moyal distributions for pygama
"""

from math import erfc

import numba as nb
import numpy as np

from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.utils import numba_math_defaults as nb_defaults
from pygama.utils import numba_math_defaults_kwargs as nb_kwargs


@nb.njit(**nb_kwargs)
def nb_moyal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    Normalised Moyal probability distribution function, w/ args: mu, sigma. Its support is :math:`x\in\mathbb{R}`
    It computes:


    .. math::
        pdf(x, \mu, \sigma) = \frac{\exp\left(-\left(\frac{x-\mu}{\sigma}+\exp\left(-\frac{x-\mu}{\sigma}\right)\right)/2\right)}{\sigma\sqrt{2\pi}}


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
    """

    y = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        y[i] = (x[i] - mu) / sigma
        y[i] = np.exp(-1 * (y[i] + np.exp(-y[i])) / 2.0) / np.sqrt(2.0 * np.pi) / sigma
    return y


@nb.njit(**nb_kwargs)
def nb_moyal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    r"""
    Normalised Moyal cumulative distribution, w/ args: mu, sigma.
    Its support is :math:`x\in\mathbb{R}`
    It computes:


    .. math::
        cdf(x, \mu, \sigma) = \text{erfc}\left(\exp\left(-\frac{x-\mu}{2\sigma}\right)/\sqrt{2}\right)


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
    """

    y = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        y[i] = (x[i] - mu) / sigma
        y[i] = erfc(np.exp(-1 * y[i] / 2) / np.sqrt(2))
    return y


@nb.njit(**nb_defaults(parallel=False))
def nb_moyal_scaled_pdf(
    x: np.ndarray, area: float, mu: float, sigma: float
) -> np.ndarray:
    r"""
    Scaled Moyal probability density function, w/ args: mu, sigma, area.
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
    """

    return area * nb_moyal_pdf(x, mu, sigma)


@nb.njit(**nb_defaults(parallel=False))
def nb_moyal_scaled_cdf(
    x: np.ndarray, area: float, mu: float, sigma: float
) -> np.ndarray:
    r"""
    Moyal cdf scaled by the area, used for extended binned fits
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
    """

    return area * nb_moyal_cdf(x, mu, sigma)


class MoyalGen(PygamaContinuous):

    def __init__(self, *args, **kwargs):
        self.x_lo = -1 * np.inf
        self.x_hi = np.inf
        super().__init__(*args, **kwargs)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        x.flags.writeable = True
        return nb_moyal_pdf(x, 0, 1)

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        x.flags.writeable = True
        return nb_moyal_cdf(x, 0, 1)

    def get_pdf(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return nb_moyal_pdf(x, mu, sigma)

    def get_cdf(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return nb_moyal_cdf(x, mu, sigma)

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
        return np.diff(nb_moyal_scaled_cdf(np.array([x_lo, x_hi]), area, mu, sigma))[
            0
        ], nb_moyal_scaled_pdf(x, area, mu, sigma)

    def cdf_ext(
        self, x: np.ndarray, area: float, mu: float, sigma: float
    ) -> np.ndarray:
        return nb_moyal_scaled_cdf(x, area, mu, sigma)

    def required_args(self) -> tuple[str, str]:
        return "mu", "sigma"


moyal = MoyalGen(name="moyal")
