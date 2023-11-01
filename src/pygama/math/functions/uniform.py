"""
Uniform distributions for pygama
"""

import sys


import numba as nb
import numpy as np
from numba import prange

from pygama.math.functions.pygama_continuous import pygama_continuous 


kwd = {"parallel": False, "fastmath": True}
kwd_parallel = {"parallel": True, "fastmath": True}


@nb.njit(**kwd_parallel)
def nb_uniform_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    r"""
    Normalised uniform probability density function, w/ args: a, b. Its range of support is :math:`x\in[a,b]`. If :math:`a=np.inf, b=np.inf` then the function
    computes :math:`a=` :func:`np.amin(x)`, :math:`b=` :func:`np.amax(x)`. The pdf is computed as:


    .. math::
        pdf(x, a, b) = \begin{cases} \frac{1}{b-a} \quad , a\leq x\leq b \\ 0 \quad , \text{otherwise} \end{cases}


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    a
        The lower edge of the distribution
    b
        The upper edge of the distribution
    """

    if a == np.inf and b == np.inf:
        a = np.amin(x)
        b = np.amax(x)
    b = a+b # gives dist on [a, a+b] like scipy's does
    w = b-a
    p = np.empty_like(x, np.float64)
    for i in prange(x.shape[0]):
        if a <= x[i] <= b:
            p[i] = 1/w
        else:
            p[i] = 0
    return p


@nb.njit(**kwd_parallel)
def nb_uniform_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    r"""
    Normalised uniform cumulative distribution, w/ args: a, b. Its range of support is :math:`x\in[a,b]`. If :math:`a=np.inf, b=np.inf` then the function
    computes :math:`a=` :func:`np.amin(x)`, :math:`b=` :func:`np.amax(x)`. The cdf is computed as:


    .. math::
        cdf(x, a, b) = \begin{cases} 0 \quad , x<a \\ \frac{x-a}{b-a} \quad , a\leq x\leq b  \\ 1 \quad , x>b \end{cases}


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    a
        The lower edge of the distribution
    b
        The upper edge of the distribution
    """

    if a == np.inf and b == np.inf:
        a = np.amin(x)
        b = np.amax(x)
    b = a+b # gives dist on [a, a+b] like scipy's does
    w = b-a
    p = np.empty_like(x, np.float64)
    for i in prange(x.shape[0]):
        if a <= x[i]:
            if x[i] <= b:
                p[i] = (x[i]-a)/w
            else: 
                p[i] = 1
        else:
            p[i] = 0
    return p


@nb.njit(**kwd)
def nb_uniform_scaled_pdf(x: np.ndarray, a: float, b: float, area: float) -> np.ndarray:
    r"""
    Scaled uniform probability distribution, w/ args: a, b.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        Input data
    a
        The lower edge of the distribution
    b
        The upper edge of the distribution
    area
        The number of counts in the signal
    """ 

    return area * nb_uniform_pdf(x, a, b)


@nb.njit(**kwd)
def nb_uniform_scaled_cdf(x: np.ndarray, a: float, b: float, area: float) -> np.ndarray:
    r"""
    Uniform cdf scaled by the area, used for extended binned fits 
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        Input data
    a
        The lower edge of the distribution
    b
        The upper edge of the distribution
    area
        The number of counts in the signal
    """ 
    
    return area * nb_uniform_cdf(x, a, b)


class uniform_gen(pygama_continuous):

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        return nb_uniform_pdf(x, 0, 1)
    def _cdf(self, x: np.ndarray) -> np.ndarray:
        return nb_uniform_cdf(x, 0, 1)

    def get_pdf(self, x: np.ndarray, a: float = np.inf, b: float = np.inf) -> np.ndarray:
        return nb_uniform_pdf(x, a, b) 
    def get_cdf(self, x: np.ndarray, a: float = np.inf, b: float = np.inf) -> np.ndarray:
        return nb_uniform_cdf(x, a, b)

    def pdf_norm(self, x: np.ndarray, x_lower: float, x_upper: float, a: float, b: float) -> np.ndarray: 
        return self._pdf_norm(x, x_lower, x_upper, a, b)
    def cdf_norm(self, x: np.ndarray, x_lower: float, x_upper: float, a: float, b: float) -> np.ndarray: 
        return self._cdf_norm(x, x_lower, x_upper, a, b)

    def pdf_ext(self, x: np.ndarray, area: float, x_lo: float, x_hi: float, a: float = np.inf, b: float = np.inf) -> np.ndarray:
        return np.sum(nb_uniform_scaled_cdf(np.array([x_lo, x_hi]), a, b, area)), nb_uniform_scaled_pdf(x, a, b, area)
    def cdf_ext(self, x: np.ndarray, area: float, a: float = np.inf, b: float = np.inf) -> np.ndarray:
        return nb_uniform_scaled_cdf(x, a, b, area)

    def required_args(self) -> tuple[str, str]:
        return "a", "b"

uniform = uniform_gen(a=0.0, b=1.0, name='uniform')