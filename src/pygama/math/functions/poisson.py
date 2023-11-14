"""
Poisson distributions for pygama
"""
import sys

import numba as nb
import numpy as np
from numba import prange

from scipy.stats import rv_discrete

kwd = {"parallel": False, "fastmath": True}
kwd_parallel = {"parallel": True, "fastmath": True}

@nb.njit(**kwd)
def factorial(nn):
    res = 1
    for ii in nb.prange(2, nn + 1):
        res *= ii
    return res

@nb.njit(**kwd_parallel)
def nb_poisson_pmf(x: np.ndarray, lamb: float, mu: int) -> np.ndarray:
    r"""
    Normalised Poisson distribution, w/ args: lamb, mu.
    The range of support is :math:`\mathbb{N}`, with :math:`lamb` :math:`\in (0,\infty)`, :math:`\mu \in \mathbb{N}`
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    .. math::
        pmf(x, \lambda, \mu) = \frac{\lambda^{x-\mu} e^{-\lambda}}{(x-\mu)!}

    Parameters
    ----------
    x : integer array-like
        The input data
    lamb
        The rate
    mu
        Amount to shift the distribution
    """
    
    y = np.empty_like(x, dtype = np.float64)
    for i in nb.prange(x.shape[0]): 
        y[i] = x[i] - mu
        if y[i] < 0:
            y[i] = 0
        else:
            y[i] = lamb**y[i] * np.exp(-lamb) / factorial(int(y[i]))
    return y


@nb.njit(**kwd_parallel)
def nb_poisson_cdf(x: np.ndarray, lamb: float, mu: int) -> np.ndarray:
    r"""
    Normalised Poisson cumulative distribution, w/ args: lamb, mu.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    .. math::
        cdf(x, \lambda, \mu) = e^{-\lambda}\sum_{j=0}^{\lfloor x-\mu \rfloor}\frac{\lambda^j}{j!}

    Parameters
    ----------
    x : integer array-like
        The input data
    lamb
        The rate
    mu
        Amount to shift the distribution
    """

    y = np.empty_like(x, dtype = np.float64)
    for i in nb.prange(x.shape[0]): 
        y[i] = x[i] - mu
        z = 0
        for j in nb.prange(1, np.floor(y[i])+2):
            j -= 1
            z += lamb**j / factorial(j)
        y[i] =  z*np.exp(-lamb)
    return y



@nb.njit(**kwd)
def nb_poisson_scaled_pmf(x: np.ndarray, lamb: float, mu: int, area: float) -> np.ndarray:
    r"""
    Scaled Poisson probability distribution, w/ args: lamb, mu.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : integer array-like
        The input data
    lamb
        The rate
    mu
        Amount to shift the distribution
    area
        The number of counts in the signal
    """ 

    return area * nb_poisson_pmf(x, lamb, mu)


@nb.njit(**kwd)
def nb_poisson_scaled_cdf(x: np.ndarray, lamb: float, mu: int, area: float) -> np.ndarray:
    r"""
    Poisson cdf scaled by the number of signal counts for extended binned fits 
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : integer array-like
        The input data
    lamb
        The rate
    mu
        Amount to shift the distribution
    area
        The number of counts in the signal
    """ 
    
    return area * nb_poisson_cdf(x, lamb, mu)

    
class poisson_gen(rv_discrete):

    def __init__(self, *args, **kwargs):
        self.x_lo = 0
        self.x_hi = np.inf
        super().__init__()

    def set_x_lo(self, x_lo):
        self.x_lo = x_lo

    def set_x_hi(self, x_hi):
        self.x_hi = x_hi

    def _pmf(self, x, lamb):
        x.flags.writeable = True
        return nb_poisson_pmf(x, lamb[0], 0)
    def _cdf(self, x, lamb):
        x.flags.writeable = True
        return nb_poisson_cdf(x, lamb[0], 0)

    def get_pmf(self, x, lamb, mu):
        return nb_poisson_pmf(x, lamb, mu)
    def get_cdf(self, x, lamb, mu):
        return nb_poisson_cdf(x, lamb, mu)

    def pmf_ext(self, x, area, lamb, mu):
        return nb_poisson_scaled_cdf(np.array([self.x_hi]), lamb, mu, area)[0]-nb_poisson_scaled_cdf(np.array([self.x_lo]), lamb, mu, area)[0], nb_poisson_scaled_pmf(x, lamb, mu, area)
    def cdf_ext(self, x, area, lamb, mu):
        return nb_poisson_scaled_cdf(x, lamb, mu, area)

    def required_args(self) -> tuple[str, str]:
        return "lamb", "mu"

poisson = poisson_gen(name='poisson')