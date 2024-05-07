"""
Poisson distributions for pygama
"""

import numba as nb
import numpy as np
from scipy.stats import rv_discrete

from pygama.utils import numba_math_defaults as nb_defaults
from pygama.utils import numba_math_defaults_kwargs as nb_kwargs


@nb.njit(**nb_defaults(parallel=False))
def factorial(nn):
    res = 1
    for ii in nb.prange(2, nn + 1):
        res *= ii
    return res


@nb.njit(**nb_kwargs)
def nb_poisson_pmf(x: np.ndarray, mu: int, lamb: float) -> np.ndarray:
    r"""
    Normalised Poisson distribution, w/ args: mu, lamb.
    The range of support is :math:`\mathbb{N}`, with :math:`lamb` :math:`\in (0,\infty)`, :math:`\mu \in \mathbb{N}`
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    .. math::
        pmf(x, \lambda, \mu) = \frac{\lambda^{x-\mu} e^{-\lambda}}{(x-\mu)!}

    Parameters
    ----------
    x : integer array-like
        The input data
    mu
        Amount to shift the distribution
    lamb
        The rate
    """

    y = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        y[i] = x[i] - mu
        if y[i] < 0:
            y[i] = 0
        else:
            y[i] = lamb ** y[i] * np.exp(-lamb) / factorial(int(y[i]))
    return y


@nb.njit(**nb_kwargs)
def nb_poisson_cdf(x: np.ndarray, mu: int, lamb: float) -> np.ndarray:
    r"""
    Normalised Poisson cumulative distribution, w/ args: mu, lamb.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    .. math::
        cdf(x, \lambda, \mu) = e^{-\lambda}\sum_{j=0}^{\lfloor x-\mu \rfloor}\frac{\lambda^j}{j!}

    Parameters
    ----------
    x : integer array-like
        The input data
    mu
        Amount to shift the distribution
    lamb
        The rate
    """

    y = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        y[i] = x[i] - mu
        z = 0
        for j in nb.prange(1, np.floor(y[i]) + 2):
            j -= 1
            z += lamb**j / factorial(j)
        y[i] = z * np.exp(-lamb)
    return y


@nb.njit(**nb_defaults(parallel=False))
def nb_poisson_scaled_pmf(
    x: np.ndarray, area: float, mu: int, lamb: float
) -> np.ndarray:
    r"""
    Scaled Poisson probability distribution, w/ args: lamb, mu.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : integer array-like
        The input data
    area
        The number of counts in the signal
    mu
        Amount to shift the distribution
    lamb
        The rate
    """

    return area * nb_poisson_pmf(x, mu, lamb)


@nb.njit(**nb_defaults(parallel=False))
def nb_poisson_scaled_cdf(
    x: np.ndarray, area: float, mu: int, lamb: float
) -> np.ndarray:
    r"""
    Poisson cdf scaled by the number of signal counts for extended binned fits
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : integer array-like
        The input data
    area
        The number of counts in the signal
    mu
        Amount to shift the distribution
    lamb
        The rate
    """

    return area * nb_poisson_cdf(x, mu, lamb)


class PoissonGen(rv_discrete):

    def __init__(self, *args, **kwargs):
        self.x_lo = 0
        self.x_hi = np.inf
        super().__init__(*args, **kwargs)

    def _pmf(self, x: np.array, mu: int, lamb: float) -> np.array:
        x.flags.writeable = True
        return nb_poisson_pmf(x, mu[0], lamb[0])

    def _cdf(self, x: np.array, mu: int, lamb: float) -> np.array:
        x.flags.writeable = True
        return nb_poisson_cdf(x, mu[0], lamb[0])

    def get_pmf(self, x: np.array, mu: int, lamb: float) -> np.array:
        return nb_poisson_pmf(x, mu, lamb)

    def get_cdf(self, x: np.array, mu: int, lamb: float) -> np.array:
        return nb_poisson_cdf(x, mu, lamb)

    def pmf_ext(
        self, x: np.array, x_lo: float, x_hi: float, area: float, mu: int, lamb: float
    ) -> np.array:
        return np.diff(nb_poisson_scaled_cdf(np.array([x_lo, x_hi]), area, mu, lamb))[
            0
        ], nb_poisson_scaled_pmf(x, area, mu, lamb)

    def cdf_ext(self, x: np.array, area: float, mu: int, lamb: float) -> np.array:
        return nb_poisson_scaled_cdf(x, area, mu, lamb)

    def required_args(self) -> tuple[str, str]:
        return "mu", "lamb"


poisson = PoissonGen(a=0, name="poisson")
