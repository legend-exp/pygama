"""
Crystal ball distributions for Pygama
"""

from math import erf

import numba as nb
import numpy as np

from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.utils import numba_math_defaults as nb_defaults
from pygama.utils import numba_math_defaults_kwargs as nb_kwargs


@nb.njit(**nb_kwargs)
def nb_crystal_ball_pdf(
    x: np.ndarray, mu: float, sigma: float, beta: float, m: float
) -> np.ndarray:
    r"""
    PDF of a power-law tail plus gaussian. Its range of support is :math:`x\in\mathbb{R}, \beta>0, m>1`. It computes:


    .. math::
        pdf(x, \beta, m, \mu, \sigma) =  \begin{cases}NA(B-\frac{x-\mu}{\sigma})^{-m} \quad \frac{x-\mu}{\sigma}\leq -\beta \\ Ne^{-(\frac{x-\mu}{\sigma})^2/2} \quad \frac{x-\mu}{\sigma}>-\beta\end{cases}


    Where


    .. math::
        A =  \frac{m^m}{\beta^m}e^{-\beta^2/2} \\
        B = \frac{m}{\beta}-\beta \\
        n =  \frac{1}{\sigma \frac{m e^{-\beta^2/2}}{\beta(m-1)} + \sigma \sqrt{\frac{\pi}{2}}\left(1+\text{erf}\left(\frac{\beta}{\sqrt{2}}\right)\right)}


    As a Numba vectorized function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    beta
        The point where the pdf changes from power-law to Gaussian
    m
        The power of the power-law tail
    """

    if (beta <= 0) or (m <= 1):
        raise ValueError("beta must be greater than 0, and m must be greater than 1")

    # Define some constants to calculate the function
    const_a = (m / np.abs(beta)) ** m * np.exp(-1 * beta**2 / 2.0)
    const_b = m / np.abs(beta) - np.abs(beta)

    n = 1.0 / (
        m / np.abs(beta) / (m - 1) * np.exp(-(beta**2) / 2.0)
        + np.sqrt(np.pi / 2) * (1 + erf(np.abs(beta) / np.sqrt(2.0)))
    )

    y = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        # Shift the distribution
        y[i] = (x[i] - mu) / sigma
        # Check if it is powerlaw
        if y[i] <= -1 * beta:
            y[i] = n * const_a * (const_b - y[i]) ** (-1 * m) / sigma
        # If it isn't power law, then it Gaussian
        else:
            y[i] = n * np.exp(-1 * y[i] ** 2 / 2) / sigma

    return y


@nb.njit(**nb_kwargs)
def nb_crystal_ball_cdf(
    x: np.ndarray, mu: float, sigma: float, beta: float, m: float
) -> np.ndarray:
    r"""
    CDF for power-law tail plus gaussian. Its range of support is :math:`x\in\mathbb{R}, \beta>0, m>1`. It computes:


    .. math::
        cdf(x, \beta, m,  \mu, \sigma)= \begin{cases}  NA\sigma\frac{(B-\frac{x-\mu}{\sigma})^{1-m}}{m-1} \quad , \frac{x-\mu}{\sigma} \leq -\beta \\ NA\sigma\frac{(B+\beta)^{1-m}}{m-1} + n\sigma \sqrt{\frac{\pi}{2}}\left(\text{erf}\left(\frac{x-\mu}{\sigma \sqrt{2}}\right)+\text{erf}\left(\frac{\beta}{\sqrt{2}}\right)\right)  \quad , \frac{x-\mu}{\sigma} >  -\beta \end{cases}


    Where


    .. math::
        A =  \frac{m^m}{\beta^m}e^{-\beta^2/2} \\
        B = \frac{m}{\beta}-\beta \\
        n =  \frac{1}{\sigma \frac{m e^{-\beta^2/2}}{\beta(m-1)} + \sigma \sqrt{\frac{\pi}{2}}\left(1+\text{erf}\left(\frac{\beta}{\sqrt{2}}\right)\right)}


    As a Numba vectorized function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    beta
        The point where the cdf changes from power-law to Gaussian
    m
        The power of the power-law tail
    """

    if (beta <= 0) or (m <= 1):
        raise ValueError("beta must be greater than 0, and m must be greater than 1")
    # Define some constants to calculate the function
    const_a = (m / np.abs(beta)) ** m * np.exp(-1 * beta**2 / 2.0)
    const_b = m / np.abs(beta) - np.abs(beta)

    # Calculate the normalization constant
    n = 1.0 / (
        (np.sqrt(np.pi / 2) * (erf(beta / np.sqrt(2)) + 1))
        + ((const_a * (const_b + beta) ** (1 - m)) / (m - 1))
    )

    y = np.empty_like(x, dtype=np.float64)

    # Check if it is in the power law part
    for i in nb.prange(x.shape[0]):
        # Shift the distribution
        y[i] = (x[i] - mu) / sigma
        if y[i] <= -1 * beta:
            y[i] = n * const_a * ((const_b - y[i]) ** (1 - m)) / (m - 1)

        # If it isn't in the power law, then it is Gaussian
        else:
            y[i] = const_a * n * ((const_b + beta) ** (1 - m)) / (m - 1) + n * np.sqrt(
                np.pi / 2
            ) * (erf(beta / np.sqrt(2)) + erf(y[i] / np.sqrt(2)))
    return y


@nb.njit(**nb_defaults(parallel=False))
def nb_crystal_ball_scaled_pdf(
    x: np.ndarray, area: float, mu: float, sigma: float, beta: float, m: float
) -> np.ndarray:
    r"""
    Scaled PDF of a power-law tail plus gaussian.
    As a Numba vectorized function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    area
        The number of counts in the distribution
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    beta
        The point where the pdf changes from power-law to Gaussian
    m
        The power of the power-law tail
    """

    return area * nb_crystal_ball_pdf(x, mu, sigma, beta, m)


@nb.njit(**nb_defaults(parallel=False))
def nb_crystal_ball_scaled_cdf(
    x: np.ndarray, area: float, mu: float, sigma: float, beta: float, m: float
) -> np.ndarray:
    r"""
    Scaled CDF for power-law tail plus gaussian. Used for extended binned fits.
    As a Numba vectorized function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    area
        The number of counts in the distribution
    mu
        The amount to shift the distribution
    sigma
        The amount to scale the distribution
    beta
        The point where the cdf changes from power-law to Gaussian
    m
        The power of the power-law tail
    """

    return area * nb_crystal_ball_cdf(x, mu, sigma, beta, m)


class CrystalBallGen(PygamaContinuous):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_lo = -1 * np.inf
        self.x_hi = np.inf

    def _pdf(
        self, x: np.ndarray, mu: float, sigma: float, beta: float, m: float
    ) -> np.ndarray:
        x.flags.writeable = True
        return nb_crystal_ball_pdf(x, mu[0], sigma[0], beta[0], m[0])

    def _cdf(
        self, x: np.ndarray, mu: float, sigma: float, beta: float, m: float
    ) -> np.ndarray:
        x.flags.writeable = True
        return nb_crystal_ball_cdf(x, mu[0], sigma[0], beta[0], m[0])

    def get_pdf(
        self, x: np.ndarray, mu: float, sigma: float, beta: float, m: float
    ) -> np.ndarray:
        return nb_crystal_ball_pdf(x, mu, sigma, beta, m)

    def get_cdf(
        self, x: np.ndarray, mu: float, sigma: float, beta: float, m: float
    ) -> np.ndarray:
        return nb_crystal_ball_cdf(x, mu, sigma, beta, m)

    def pdf_norm(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        beta: float,
        m: float,
    ) -> np.ndarray:
        return self._pdf_norm(x, x_lo, x_hi, mu, sigma, beta, m)

    def cdf_norm(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        beta: float,
        m: float,
    ) -> np.ndarray:
        return self._cdf_norm(x, x_lo, x_hi, mu, sigma, beta, m)

    def pdf_ext(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        area: float,
        mu: float,
        sigma: float,
        beta: float,
        m: float,
    ) -> np.ndarray:
        return np.diff(
            nb_crystal_ball_scaled_cdf(np.array([x_lo, x_hi]), area, mu, sigma, beta, m)
        )[0], nb_crystal_ball_scaled_pdf(x, area, mu, sigma, beta, m)

    def cdf_ext(
        self, x: np.ndarray, area: float, mu: float, sigma: float, beta: float, m: float
    ) -> np.ndarray:
        return nb_crystal_ball_scaled_cdf(x, area, mu, sigma, beta, m)

    def required_args(self) -> tuple[str, str, str, str]:
        return "mu", "sigma", "beta", "m"


crystal_ball = CrystalBallGen(name="crystal_ball")
