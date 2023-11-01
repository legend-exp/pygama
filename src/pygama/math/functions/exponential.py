"""
Exponential distributions for pygama
"""

import sys

import numba as nb
import numpy as np

from pygama.math.functions.pygama_continuous import pygama_continuous

kwd = {"parallel": False, "fastmath": True}
kwd_parallel = {"parallel": True, "fastmath": True}


@nb.njit(**kwd_parallel)
def nb_exponential_pdf(x: np.ndarray, lamb: float, mu: float, sigma: float) -> np.ndarray:
    r"""
    Normalised exponential probability density distribution, w/ args: lamb, mu, sigma. Its range of support is :math:`x\in[0,\infty), \lambda>0`. 
    It computes:


    .. math::
        pdf(x, \lambda, \mu, \sigma) = \begin{cases} \lambda e^{-\lambda\frac{x-\mu}{\sigma}} \quad , \frac{x-\mu}{\sigma}\geq 0 \\ 0 \quad , \frac{x-\mu}{\sigma}<0 \end{cases} 


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        The input data
    lamb
        The rate
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    """

    y = np.empty_like(x, dtype = np.float64)
    for i in nb.prange(x.shape[0]):
        y[i] = (x[i]-mu)/sigma
        if y[i] <0 :
            y[i] = 0
        else:
            y[i] = (lamb*np.exp(-1*lamb*y[i]))/sigma
    return y


@nb.njit(**kwd_parallel)
def nb_exponential_cdf(x: np.ndarray, lamb: float, mu: float, sigma: float) -> np.ndarray:
    r"""
    Normalised exponential cumulative distribution, w/ args: lamb, mu, sigma. Its range of support is :math:`x\in[0,\infty), \lambda>0`. 
    It computes:


    .. math::
        cdf(x, \lambda, \mu, \sigma) = \begin{cases}  1-e^{-\lambda\frac{x-\mu}{\sigma}} \quad , \frac{x-\mu}{\sigma} > 0 \\ 0 \quad , \frac{x-\mu}{\sigma}\leq 0 \end{cases} 


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        The input data
    lamb
        The rate
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    """

    y = np.empty_like(x, dtype = np.float64)
    for i in nb.prange(x.shape[0]):
        y[i] = (x[i]-mu)/sigma
        if y[i] <= 0:
            y[i] = 0 
        else:
            y[i] = (1-np.exp(-1*lamb*y[i]))
    return y


@nb.njit(**kwd)
def nb_exponential_scaled_pdf(x: np.ndarray, lamb: float, mu: float, sigma: float, area: float) -> np.ndarray:
    r"""
    Scaled exponential probability distribution, w/ args: lamb, mu, sigma, area.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        Input data
    lamb
        The rate
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    area
        The number of counts in the signal
    """ 

    return area * nb_exponential_pdf(x, lamb, mu, sigma)


@nb.njit(**kwd)
def nb_exponential_scaled_cdf(x: np.ndarray, lamb: float, mu: float, sigma: float, area: float) -> np.ndarray:
    r"""
    Exponential cdf scaled by the area, used for extended binned fits 
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        Input data
    lamb
        The rate
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    area
        The number of counts in the signal
    """ 
    
    return area * nb_exponential_cdf(x, lamb, mu, sigma)


class exponential_gen(pygama_continuous):

    def _pdf(self, x: np.ndarray, lamb: float) -> np.ndarray:
        x.flags.writeable = True
        return nb_exponential_pdf(x, lamb[0], 0, 1)
    def _cdf(self, x: np.ndarray, lamb: float) -> np.ndarray:
        x.flags.writeable = True
        return nb_exponential_cdf(x, lamb[0], 0, 1)

    def get_pdf(self, x: np.ndarray, lamb: float, mu: float, sigma: float) -> np.ndarray:
        return nb_exponential_pdf(x, lamb, mu, sigma) 
    def get_cdf(self, x: np.ndarray, lamb: float, mu: float, sigma: float) -> np.ndarray:
        return nb_exponential_cdf(x, lamb, mu, sigma)

    # needed so that we can hack iminuit's introspection to function parameter names... unless
    def pdf_norm(self, x: np.ndarray, x_lower: float, x_upper: float,  lamb: float, mu: float, sigma: float) -> np.ndarray: 
        return self._pdf_norm(x, x_lower, x_upper, lamb, mu, sigma)
    def cdf_norm(self, x: np.ndarray, x_lower: float, x_upper: float, lamb: float, mu: float, sigma: float) -> np.ndarray: 
        return self._cdf_norm(x, x_lower, x_upper, lamb, mu, sigma)

    def pdf_ext(self, x: np.ndarray, area: float, x_lo: float, x_hi: float, lamb: float, mu: float, sigma: float) -> np.ndarray:
        return nb_exponential_scaled_cdf(np.array([x_hi]), lamb, mu, sigma, area)[0]-nb_exponential_scaled_cdf(np.array([x_lo]), lamb, mu, sigma, area)[0], nb_exponential_scaled_pdf(x, lamb, mu, sigma, area)
    def cdf_ext(self, x: np.ndarray, area: float, lamb: float, mu: float, sigma: float) -> np.ndarray:
        return nb_exponential_scaled_cdf(x, lamb, mu, sigma, area)
    
    def required_args(self) -> tuple[str, str, str]:
        return "lambda", "mu", "sigma"

exponential = exponential_gen(a=0.0, name='exponential')