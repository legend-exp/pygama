import sys


import numba as nb
import numpy as np
from numba import prange

from pygama.math.functions.pygama_continuous import pygama_continuous 


kwd = {"parallel": False, "fastmath": True}
kwd_parallel = {"parallel": True, "fastmath": True}


@nb.njit(**kwd_parallel)
def nb_linear_pdf(x: np.ndarray, x_lower: float, x_upper: float, m: float, b: float) -> np.ndarray:
    r"""
    Normalised linear probability density function, w/ args: m, b. Its range of support is :math:`x\in(x_{lower},x_{upper})`. 
    If :math:`x_{lower} = np.inf` and :math:`x_{upper} = np.inf`, then the function takes  :math:`x_{upper} = :func:`np.min(x)` and :math:`x_{upper} = :func:`np.amax(x)`
    It computes:


    .. math::
        pdf(x, x_{lower}, x_{upper}, m, b) = \frac{mx+b}{\frac{m}{2}(x_{upper}^2-x_{lower}^2)+b(x_{upper}-x_{lower})}


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lower
        The lower bound of the distribution
    x_upper
        The upper bound of the distribution
    m
        The slope of the linear part
    b
        The y-intercept of the linear part
    """
    norm = (m/2)*(x_upper**2 - x_lower**2) + b*(x_upper-x_lower)

    result = np.empty_like(x, np.float64)
    for i in prange(x.shape[0]):
        result[i] = (m*x[i] + b)/norm
    return result


@nb.njit(**kwd_parallel)
def nb_linear_cdf(x: np.ndarray, x_lower: float, x_upper: float, m: float, b: float) -> np.ndarray:
    r"""
    Normalised linear cumulative density function, w/ args: m, b. Its range of support is :math:`x\in(x_{lower},x_{upper})`. 
    If :math:`x_{lower} = np.inf` and :math:`x_{upper} = np.inf`, then the function takes  :math:`x_{upper} = :func:`np.min(x)` and :math:`x_{upper} = :func:`np.amax(x)`
    It computes:


    .. math::
        cdf(x, x_{lower}, x_{upper}, m, b) = \frac{\frac{m}{2}(x^2-x_{lower}^2)+b(x-x_{lower})}{\frac{m}{2}(x_{upper}^2-x_{lower}^2)+b(x_{upper}-x_{lower})}


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lower
        The lower bound of the distribution
    x_upper
        The upper bound of the distribution
    m
        The slope of the linear part
    b
        The y-intercept of the linear part
    """
    norm = (m/2)*(x_upper**2 - x_lower**2) + b*(x_upper-x_lower)

    result = np.empty_like(x, np.float64)
    for i in prange(x.shape[0]):
        result[i] = (m/2*(x[i]**2-x_lower**2) + b*(x[i]-x_lower))/norm
    return result


@nb.njit(**kwd)
def nb_linear_scaled_pdf(x: np.ndarray, x_lower: float, x_upper: float, m: float, b: float, area: float) -> np.ndarray:
    r"""
    Scaled linear probability distribution, w/ args: m, b.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lower
        The lower bound of the distribution
    x_upper
        The upper bound of the distribution
    m
        The slope of the linear part
    b
        The y-intercept of the linear part
    area
        The number of counts in the signal
    """ 

    return area * nb_linear_pdf(x, x_lower, x_upper, m, b)


@nb.njit(**kwd)
def nb_linear_scaled_cdf(x: np.ndarray, x_lower: float, x_upper: float, m: float, b: float, area: float) -> np.ndarray:
    r"""
    Linear cdf scaled by the area for extended binned fits 
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lower
        The lower bound of the distribution
    x_upper
        The upper bound of the distribution
    m
        The slope of the linear part
    b
        The y-intercept of the linear part
    area
        The number of counts in the signal
    """ 
    
    return area * nb_linear_cdf(x, x_lower, x_upper, m, b)


class linear_gen(pygama_continuous):
    
    def __init__(self, *args, **kwargs):
        self.x_lo = None
        self.x_hi = None
        super().__init__(self)

    def _argcheck(self, x_lower, x_upper, m, b):
        return True

    def _pdf(self, x: np.ndarray, x_lower: float, x_upper: float, m, b) -> np.ndarray:
        x.flags.writeable = True
        return nb_linear_pdf(x, x_lower[0], x_upper[0],  m[0], b[0])
    def _cdf(self, x: np.ndarray, x_lower: float, x_upper: float, m, b) -> np.ndarray:
        x.flags.writeable = True
        return nb_linear_cdf(x, x_lower[0], x_upper[0], m[0], b[0])

    def get_pdf(self, x: np.ndarray, x_lower: float, x_upper: float, m: float, b: float) -> np.ndarray:
        return nb_linear_pdf(x, x_lower, x_upper, m, b) 
    def get_cdf(self, x: np.ndarray, x_lower: float, x_upper: float, m: float, b: float) -> np.ndarray:
        return nb_linear_cdf(x, x_lower, x_upper, m, b)

    # Because this function is already normalized over its limited support, we need to alias get_pdf as pdf_norm 
    def pdf_norm(self, x: np.ndarray, x_lower: float, x_upper: float, m: float, b: float) -> np.ndarray:
        return nb_linear_pdf(x, x_lower, x_upper, m, b) 
    def cdf_norm(self, x: np.ndarray, x_lower: float, x_upper: float, m: float, b: float) -> np.ndarray:
        return nb_linear_cdf(x, x_lower, x_upper, m, b)

    def pdf_ext(self, x: np.ndarray, area: float, x_lower: float, x_upper: float, m: float, b: float) -> np.ndarray:
        return np.diff(nb_linear_scaled_cdf(np.array([self.x_lo, self.x_hi]), x_lower, x_upper, m, b, area))[0], nb_linear_scaled_pdf(x, x_lower, x_upper, m, b, area)
    def cdf_ext(self, x: np.ndarray, area: float, x_lower: float, x_upper: float, m: float, b: float) -> np.ndarray:
        return nb_linear_scaled_cdf(x, x_lower, x_upper, m, b, area)

    def required_args(self) -> tuple[str, str, str, str]:
        return  "x_lower", "x_upper", "m", "b"

linear = linear_gen(name='linear')