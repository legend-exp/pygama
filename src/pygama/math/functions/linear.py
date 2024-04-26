import numba as nb
import numpy as np
from numba import prange

from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.utils import numba_math_defaults as nb_defaults
from pygama.utils import numba_math_defaults_kwargs as nb_kwargs


@nb.njit(**nb_kwargs)
def nb_linear_pdf(
    x: np.ndarray, x_lo: float, x_hi: float, m: float, b: float
) -> np.ndarray:
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
    x_lo
        The lower bound of the distribution
    x_hi
        The upper bound of the distribution
    m
        The slope of the linear part
    b
        The y-intercept of the linear part
    """
    norm = (m / 2) * (x_hi**2 - x_lo**2) + b * (x_hi - x_lo)

    result = np.empty_like(x, np.float64)
    for i in prange(x.shape[0]):
        result[i] = (m * x[i] + b) / norm
    return result


@nb.njit(**nb_kwargs)
def nb_linear_cdf(
    x: np.ndarray, x_lo: float, x_hi: float, m: float, b: float
) -> np.ndarray:
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
    x_lo
        The lower bound of the distribution
    x_hi
        The upper bound of the distribution
    m
        The slope of the linear part
    b
        The y-intercept of the linear part
    """
    norm = (m / 2) * (x_hi**2 - x_lo**2) + b * (x_hi - x_lo)

    result = np.empty_like(x, np.float64)
    for i in prange(x.shape[0]):
        result[i] = (m / 2 * (x[i] ** 2 - x_lo**2) + b * (x[i] - x_lo)) / norm
    return result


@nb.njit(**nb_defaults(parallel=False))
def nb_linear_scaled_pdf(
    x: np.ndarray, x_lo: float, x_hi: float, area: float, m: float, b: float
) -> np.ndarray:
    r"""
    Scaled linear probability distribution, w/ args: m, b.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lo
        The lower bound of the distribution
    x_hi
        The upper bound of the distribution
    area
        The number of counts in the signal
    m
        The slope of the linear part
    b
        The y-intercept of the linear part
    """

    return area * nb_linear_pdf(x, x_lo, x_hi, m, b)


@nb.njit(**nb_defaults(parallel=False))
def nb_linear_scaled_cdf(
    x: np.ndarray, x_lo: float, x_hi: float, area: float, m: float, b: float
) -> np.ndarray:
    r"""
    Linear cdf scaled by the area for extended binned fits
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lo
        The lower bound of the distribution
    x_hi
        The upper bound of the distribution
    area
        The number of counts in the signal
    m
        The slope of the linear part
    b
        The y-intercept of the linear part
    """

    return area * nb_linear_cdf(x, x_lo, x_hi, m, b)


class LinearGen(PygamaContinuous):

    def __init__(self, *args, **kwargs):
        self.x_lo = None
        self.x_hi = None
        super().__init__(*args, **kwargs)

    def _argcheck(self, x_lo, x_hi, m, b):
        return True

    def _pdf(self, x: np.ndarray, x_lo: float, x_hi: float, m, b) -> np.ndarray:
        x.flags.writeable = True
        return nb_linear_pdf(x, x_lo[0], x_hi[0], m[0], b[0])

    def _cdf(self, x: np.ndarray, x_lo: float, x_hi: float, m, b) -> np.ndarray:
        x.flags.writeable = True
        return nb_linear_cdf(x, x_lo[0], x_hi[0], m[0], b[0])

    def get_pdf(
        self, x: np.ndarray, x_lo: float, x_hi: float, m: float, b: float
    ) -> np.ndarray:
        return nb_linear_pdf(x, x_lo, x_hi, m, b)

    def get_cdf(
        self, x: np.ndarray, x_lo: float, x_hi: float, m: float, b: float
    ) -> np.ndarray:
        return nb_linear_cdf(x, x_lo, x_hi, m, b)

    # Because this function is already normalized over its limited support, we need to alias get_pdf as pdf_norm
    def pdf_norm(
        self, x: np.ndarray, x_lo: float, x_hi: float, m: float, b: float
    ) -> np.ndarray:
        return nb_linear_pdf(x, x_lo, x_hi, m, b)

    def cdf_norm(
        self, x: np.ndarray, x_lo: float, x_hi: float, m: float, b: float
    ) -> np.ndarray:
        return nb_linear_cdf(x, x_lo, x_hi, m, b)

    def pdf_ext(
        self, x: np.ndarray, x_lo: float, x_hi: float, area: float, m: float, b: float
    ) -> np.ndarray:
        return np.diff(
            nb_linear_scaled_cdf(np.array([x_lo, x_hi]), x_lo, x_hi, area, m, b)
        )[0], nb_linear_scaled_pdf(x, x_lo, x_hi, area, m, b)

    def cdf_ext(
        self, x: np.ndarray, x_lo: float, x_hi: float, area: float, m: float, b: float
    ) -> np.ndarray:
        return nb_linear_scaled_cdf(x, x_lo, x_hi, area, m, b)

    def required_args(self) -> tuple[str, str, str, str]:
        return "x_lo", "x_hi", "m", "b"


linear = LinearGen(name="linear")
