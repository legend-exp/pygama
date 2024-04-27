"""
Uniform distributions for pygama
"""

import numba as nb
import numpy as np
from numba import prange

from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.utils import numba_math_defaults as nb_defaults
from pygama.utils import numba_math_defaults_kwargs as nb_kwargs


@nb.njit(**nb_kwargs)
def nb_uniform_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    r"""
    Normalised uniform probability density function, w/ args: a, b. Its range of support is :math:`x\in[a,b]`. The pdf is computed as:


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
    b = a + b  # gives dist on [a, a+b] like scipy's does
    w = b - a
    p = np.empty_like(x, np.float64)
    for i in prange(x.shape[0]):
        if a <= x[i] <= b:
            p[i] = 1 / w
        else:
            p[i] = 0
    return p


@nb.njit(**nb_kwargs)
def nb_uniform_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    r"""
    Normalised uniform cumulative distribution, w/ args: a, b. Its range of support is :math:`x\in[a,b]`. The cdf is computed as:


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

    b = a + b  # gives dist on [a, a+b] like scipy's does
    w = b - a
    p = np.empty_like(x, np.float64)
    for i in prange(x.shape[0]):
        if a <= x[i]:
            if x[i] <= b:
                p[i] = (x[i] - a) / w
            else:
                p[i] = 1
        else:
            p[i] = 0
    return p


@nb.njit(**nb_defaults(parallel=False))
def nb_uniform_scaled_pdf(x: np.ndarray, area: float, a: float, b: float) -> np.ndarray:
    r"""
    Scaled uniform probability distribution, w/ args: a, b.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        Input data
    area
        The number of counts in the signal
    a
        The lower edge of the distribution
    b
        The upper edge of the distribution
    """

    return area * nb_uniform_pdf(x, a, b)


@nb.njit(**nb_defaults(parallel=False))
def nb_uniform_scaled_cdf(x: np.ndarray, area: float, a: float, b: float) -> np.ndarray:
    r"""
    Uniform cdf scaled by the area, used for extended binned fits
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        Input data
    area
        The number of counts in the signal
    a
        The lower edge of the distribution
    b
        The upper edge of the distribution
    """

    return area * nb_uniform_cdf(x, a, b)


class UniformGen(PygamaContinuous):

    def _get_support(self, a, b):
        return a, b

    def _argcheck(self, a, b):
        return b > a

    def __init__(self, *args, **kwargs):
        self.x_lo = None
        self.x_hi = None
        super().__init__(*args, **kwargs)

    def _pdf(self, x: np.ndarray, a, b) -> np.ndarray:
        return nb_uniform_pdf(x, a[0], b[0])

    def _cdf(self, x: np.ndarray, a, b) -> np.ndarray:
        return nb_uniform_cdf(x, a[0], b[0])

    def get_pdf(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return nb_uniform_pdf(x, a, b)

    def get_cdf(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return nb_uniform_cdf(x, a, b)

    def pdf_norm(
        self, x: np.ndarray, x_lo: float, x_hi: float, a: float, b: float
    ) -> np.ndarray:
        return self._pdf_norm(x, x_lo, x_hi, a, b)

    def cdf_norm(
        self, x: np.ndarray, x_lo: float, x_hi: float, a: float, b: float
    ) -> np.ndarray:
        return self._cdf_norm(x, x_lo, x_hi, a, b)

    def pdf_ext(
        self, x: np.ndarray, x_lo: float, x_hi: float, area: float, a: float, b: float
    ) -> np.ndarray:
        return np.diff(nb_uniform_scaled_cdf(np.array([x_lo, x_hi]), area, a, b))[
            0
        ], nb_uniform_scaled_pdf(x, area, a, b)

    def cdf_ext(self, x: np.ndarray, area: float, a: float, b: float) -> np.ndarray:
        return nb_uniform_scaled_cdf(x, area, a, b)

    def required_args(self) -> tuple[str, str]:
        return "a", "b"


uniform = UniformGen(a=0.0, b=1.0, name="uniform")
