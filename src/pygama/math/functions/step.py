"""
Step distributions for pygama
"""

from math import erf

import numba as nb
import numpy as np

from pygama.math.functions.pygama_continuous import PygamaContinuous
from pygama.utils import numba_math_defaults as nb_defaults
from pygama.utils import numba_math_defaults_kwargs as nb_kwargs


@nb.njit(**nb_defaults(parallel=False))
def nb_step_int(x: float, mu: float, sigma: float, hstep: float) -> np.ndarray:
    r"""
    Integral of step function w/args mu, sigma, hstep. It computes:


    .. math::
        \int cdf(x, hstep, \mu, \sigma)\, dx = \sigma\left(\frac{x-\mu}{\sigma} + hstep \left(\frac{x-\mu}{\sigma}\mathrm{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right) + \sqrt{\frac{2}{\pi}}\exp\left(-(\frac{x-\mu}{\sigma})^2/2\right) \right)\right)


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        A single input data point
    mu
        The location of the step
    sigma
        The width of the step
    hstep
        The height of the step


    Returns
    -------
    step_int
        The cumulative integral of the step distribution at x
    """

    if sigma == 0:
        y = (x) + hstep * (x - mu)
    else:
        z = (x - mu) / sigma
        y = sigma * (
            z
            + hstep
            * (z * erf(z / np.sqrt(2)) + np.sqrt(2 / np.pi) * np.exp(-(z**2) / 2))
        )
    return y


@nb.njit(**nb_defaults(parallel=False))
def nb_unnorm_step_pdf(x: float, mu: float, sigma: float, hstep: float) -> float:
    r"""
    Unnormalised step function for use in pdfs. It computes:


    .. math::
        pdf(x, hstep, \mu, \sigma) = 1+ hstep\mathrm{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)



    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        A single input data point
    mu
        The location of the step
    sigma
        The "width" of the step, because we are using an error function to define it
    hstep
        The height of the step

    """

    invs = np.sqrt(2) * sigma
    if invs == 0:
        return 1 + hstep
    else:
        step_f = 1 + hstep * erf((x - mu) / invs)
        return step_f


@nb.njit(**nb_kwargs)
def nb_step_pdf(
    x: np.ndarray, x_lo: float, x_hi: float, mu: float, sigma: float, hstep: float
) -> np.ndarray:
    r"""
    Normalised step function w/args mu, sigma, hstep, x_lo, x_hi. Its range of support is :math:`x\in` (x_lo, x_hi). It computes:


    .. math::
        pdf(x, \mathrm{x}_\mathrm{lo}, \mathrm{x}_\mathrm{hi}, \mu, \sigma, hstep) = pdf(y=\frac{x-\mu}{\sigma}, step, \mathrm{x}_\mathrm{lo}, \mathrm{x}_\mathrm{hi}) = \frac{1+hstep\mathrm{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)}{\sigma\left[(y-y_{min}) +hstep\left(y\mathrm{erf}(\frac{y}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y^2/2}-y_{min}\mathrm{erf}(\frac{y_{min}}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y_{min}^2/2}\right)\right]}


    Where :math:`y_{max} = \frac{\mathrm{x}_\mathrm{hi} - \mu}{\sigma}, y_{min} = \frac{\mathrm{x}_\mathrm{lo} - \mu}{\sigma}`.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters
    ----------
    x
        The input data
    x_lo
        The lower range on which to normalize the step PDF, default is to normalize from min to max x values
    x_hi
        The upper range on which to normalize the step PDF
    mu
        The location of the step
    sigma
        The "width" of the step, because we are using an error function to define it
    hstep
        The height of the step
    """
    # Compute the normalization
    norm = nb_step_int(x_hi, mu, sigma, hstep) - nb_step_int(x_lo, mu, sigma, hstep)

    if norm == 0:
        # If the normalization is zero, don't waste time computing the step_pdf
        z = np.full_like(x, np.inf, dtype=np.float64)

    else:
        z = np.empty_like(x, dtype=np.float64)
        for i in nb.prange(x.shape[0]):
            step_f = nb_unnorm_step_pdf(x[i], mu, sigma, hstep)
            z[i] = step_f / norm

    return z


@nb.njit(**nb_kwargs)
def nb_step_cdf(
    x: np.ndarray, x_lo: float, x_hi: float, mu: float, sigma: float, hstep: float
) -> np.ndarray:
    r"""
    Normalized CDF for step function w/args mu, sigma, hstep, x_lo, x_hi. Its range of support is :math:`x\in` (x_lo, x_hi). It computes:


    .. math::
        cdf(x, \mathrm{x}_\mathrm{lo}, \mathrm{x}_\mathrm{hi}, \mu, \sigma, hstep) = cdf(y=\frac{x-\mu}{\sigma}, hstep, \mathrm{x}_\mathrm{lo}, \mathrm{x}_\mathrm{hi}) = \frac{(y-y_{min}) +hstep\left(y\mathrm{erf}(\frac{y}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y^2/2}-y_{min}\mathrm{erf}(\frac{y_{min}}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y_{min}^2/2}\right)}{\sigma\left[(y_{max}-y_{min}) +hstep\left(y_{max}\mathrm{erf}(\frac{y_{max}}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y_{max}^2/2}-y_{min}\mathrm{erf}(\frac{y_{min}}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y_{min}^2/2}\right)\right] }


    Where :math:`y_{max} = \frac{\mathrm{x}_\mathrm{hi} - \mu}{\sigma}, y_{min} = \frac{\mathrm{x}_\mathrm{lo} - \mu}{\sigma}`.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lo
        The lower range on which to normalize the step PDF, default is to normalize from min to max x values
    x_hi
        The upper range on which to normalize the step PDF
    mu
        The location of the step
    sigma
        The "width" of the step, because we are using an error function to define it
    hstep
        The height of the step
    """
    integral_lo = nb_step_int(x_lo, mu, sigma, hstep)
    integral_hi = nb_step_int(x_hi, mu, sigma, hstep)
    norm = integral_hi - integral_lo

    if norm == 0:
        # If the normalization is zero, return np.inf and avoid wasting time computing the integral
        cdf = np.full_like(x, np.inf, dtype=np.float64)

    else:
        integrated_pdf = np.empty_like(x, dtype=np.float64)

        for i in nb.prange(x.shape[0]):
            integrated_pdf[i] = nb_step_int(x[i], mu, sigma, hstep)
        cdf = (integrated_pdf - integral_lo) / norm  # compute (cdf(x)-cdf(x_min))/norm

    return cdf


@nb.njit(**nb_defaults(parallel=False))
def nb_step_scaled_pdf(
    x: np.ndarray,
    x_lo: float,
    x_hi: float,
    area: float,
    mu: float,
    sigma: float,
    hstep: float,
) -> np.ndarray:
    r"""
    Scaled step function pdf w/args x_lo, x_hi, area, mu, sigma, hstep.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lo
        The lower range on which to normalize the step PDF, default is to normalize from min to max x values
    x_hi
        The upper range on which to normalize the step PDF
    area
        The number to scale the distribution by
    mu
        The location of the step
    sigma
        The "width" of the step, because we are using an error function to define it
    hstep
        The height of the step
    """

    return area * nb_step_pdf(x, x_lo, x_hi, mu, sigma, hstep)


@nb.njit(**nb_defaults(parallel=False))
def nb_step_scaled_cdf(
    x: np.ndarray,
    x_lo: float,
    x_hi: float,
    area: float,
    mu: float,
    sigma: float,
    hstep: float,
) -> np.ndarray:
    r"""
    Scaled step function CDF w/args  x_lo, x_hi, area, mu, sigma, hstep. Used for extended binned fits.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x
        The input data
    x_lo
        The lower range on which to normalize the step PDF, default is to normalize from min to max x values
    x_hi
        The upper range on which to normalize the step PDF
    area
        The number to scale the distribution by
    mu
        The location of the step
    sigma
        The "width" of the step, because we are using an error function to define it
    hstep
        The height of the step
    """

    return area * nb_step_cdf(x, x_lo, x_hi, mu, sigma, hstep)


class StepGen(PygamaContinuous):

    def _argcheck(self, x_lo, x_hi, mu, sigma, hstep):
        return x_hi > x_lo

    def __init__(self, *args, **kwargs):
        self.x_lo = None
        self.x_hi = None
        super().__init__(*args, **kwargs)

    def _pdf(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        hstep: float,
    ) -> np.ndarray:
        x.flags.writeable = True
        return nb_step_pdf(x, x_lo[0], x_hi[0], mu[0], sigma[0], hstep[0])

    def _cdf(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        hstep: float,
    ) -> np.ndarray:
        x.flags.writeable = True
        return nb_step_cdf(x, x_lo[0], x_hi[0], mu[0], sigma[0], hstep[0])

    def get_pdf(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        hstep: float,
    ) -> np.ndarray:
        return nb_step_pdf(x, x_lo, x_hi, mu, sigma, hstep)

    def get_cdf(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        hstep: float,
    ) -> np.ndarray:
        return nb_step_cdf(x, x_lo, x_hi, mu, sigma, hstep)

    # Because step is only defined on a user specified range, we don't need to return a different pdf_norm, just alias get_pdf and get_cdf
    def pdf_norm(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        hstep: float,
    ) -> np.ndarray:
        return nb_step_pdf(x, x_lo, x_hi, mu, sigma, hstep)

    def cdf_norm(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        mu: float,
        sigma: float,
        hstep: float,
    ) -> np.ndarray:
        return nb_step_cdf(x, x_lo, x_hi, mu, sigma, hstep)

    def pdf_ext(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        area: float,
        mu: float,
        sigma: float,
        hstep: float,
    ) -> np.ndarray:
        return np.diff(
            nb_step_scaled_cdf(
                np.array([x_lo, x_hi]), x_lo, x_hi, area, mu, sigma, hstep
            )
        )[0], nb_step_scaled_pdf(x, x_lo, x_hi, area, mu, sigma, hstep)

    def cdf_ext(
        self,
        x: np.ndarray,
        x_lo: float,
        x_hi: float,
        area: float,
        mu: float,
        sigma: float,
        hstep: float,
    ) -> np.ndarray:
        return nb_step_scaled_cdf(x, x_lo, x_hi, area, mu, sigma, hstep)

    def required_args(self) -> tuple[str, str, str, str, str]:
        return "x_lo", "x_hi", "mu", "sigma", "hstep"


step = StepGen(name="step")
