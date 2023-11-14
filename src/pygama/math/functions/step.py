"""
Step distributions for pygama
"""

import sys
from typing import Union

import numba as nb
import numpy as np
from math import erf

from pygama.math.functions.gauss import nb_gauss
from pygama.math.functions.pygama_continuous import pygama_continuous 

kwd = {"parallel": False, "fastmath": True}
kwd_parallel = {"parallel": True, "fastmath": True}

@nb.njit(**kwd_parallel)
def nb_step_int(x: np.ndarray, mu: float, sigma: float, hstep: float) -> np.ndarray:
    r"""
    Integral of step function w/args mu, sigma, hstep. It computes: 


    .. math:: 
        \int pdf(x, hstep, \mu, \sigma)\, dx = \sigma\left(\frac{x-\mu}{\sigma} + hstep \left(\frac{x-\mu}{\sigma}\text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right) + \sqrt{\frac{2}{\pi}}\exp\left(-(\frac{x-\mu}{\sigma})^2/2\right) \right)\right)


    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters 
    ----------
    x 
        The input data
    mu 
        The location of the step
    sigma 
        The width of the step 
    hstep 
        The height of the step 


    Returns 
    -------
    step_int
        The cumulative integral of the step distribution
    """

    y = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        if sigma == 0:
            y[i] = (x[i])+hstep*(x[i]-mu)
        else:
            part1 = x[i]+hstep*(x[i]-mu)*erf((x[i]-mu)/(np.sqrt(2)*sigma)) # we don't need the (x-mu) in the first term because we only care about differences in integrals
            part2 = np.sqrt(2/np.pi)*hstep*sigma*nb_gauss(x[i],mu,sigma)
            y[i] = part1+part2
    return y


@nb.njit(**kwd)
def nb_unnorm_step_pdf(x: float,  mu: float, sigma: float, hstep: float) -> float:
    r"""
    Unnormalised step function for use in pdfs. It computes: 


    .. math:: 
        pdf(x, hstep, \mu, \sigma) = 1+ hstep\text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)



    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters 
    ----------
    x
        The input data
    mu 
        The location of the step
    sigma 
        The "width" of the step, because we are using an error function to define it
    hstep 
        The height of the step 

    """

    invs = (np.sqrt(2)*sigma)
    if invs == 0: 
        return 1+hstep
    else: 
        step_f = 1 + hstep * erf((x-mu)/invs)
        return step_f


@nb.njit(**kwd_parallel)
def nb_step_pdf(x: np.ndarray,  mu: float, sigma: float, hstep: float, lower_range: float = np.inf , upper_range: float = np.inf) -> np.ndarray:
    r"""
    Normalised step function w/args mu, sigma, hstep, lower_range, upper_range. Its range of support is :math:`x\in` (lower_range, upper_range). It computes: 


    .. math:: 
        pdf(x, hstep, \mu, \sigma , \text{lower_range}, \text{upper_range}) = pdf(y=\frac{x-\mu}{\sigma}, step, \text{lower_range}, \text{upper_range}) = \frac{1+hstep\text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)}{\sigma\left[(y-y_{min}) +hstep\left(y\text{erf}(\frac{y}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y^2/2}-y_{min}\text{erf}(\frac{y_{min}}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y_{min}^2/2}\right)\right]}


    Where :math:`y_{max} = \frac{\text{upper_range} - \mu}{\sigma}, y_{min} = \frac{\text{lower_range} - \mu}{\sigma}`. 
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.


    Parameters 
    ----------
    x
        The input data
    mu 
        The location of the step
    sigma 
        The "width" of the step, because we are using an error function to define it
    hstep 
        The height of the step 
    lower_range
        The lower range on which to normalize the step PDF, default is to normalize from min to max x values
    upper_range 
        The upper range on which to normalize the step PDF

    """
    integral = nb_step_int(np.array([lower_range, upper_range]), mu, sigma, hstep)
    norm = integral[1]-integral[0]

    z = np.empty_like(x, dtype=np.float64)
    for i in nb.prange(x.shape[0]):
        step_f = nb_unnorm_step_pdf(x[i],  mu, sigma, hstep)

        if norm == 0 :
            z[i] = np.inf
        else:
            z[i] = step_f/norm
    return z


@nb.njit(**kwd)
def nb_step_cdf(x: np.ndarray, mu: float, sigma: float, hstep: float, lower_range: float = np.inf, upper_range: float = np.inf) -> np.ndarray:
    r"""
    Normalized CDF for step function w/args mu, sigma, hstep, lower_range, upper_range. Its range of support is :math:`x\in` (lower_range, upper_range). It computes: 


    .. math::
        cdf(x, hstep, \mu, \sigma, \text{lower_range}, \text{upper_range}) = cdf(y=\frac{x-\mu}{\sigma}, hstep, \text{lower_range}, \text{upper_range}) = \frac{(y-y_{min}) +hstep\left(y\text{erf}(\frac{y}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y^2/2}-y_{min}\text{erf}(\frac{y_{min}}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y_{min}^2/2}\right)}{\sigma\left[(y_{max}-y_{min}) +hstep\left(y_{max}\text{erf}(\frac{y_{max}}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y_{max}^2/2}-y_{min}\text{erf}(\frac{y_{min}}{\sqrt{2}})+\sqrt{\frac{2}{\pi}}e^{-y_{min}^2/2}\right)\right] }


    Where :math:`y_{max} = \frac{\text{upper_range} - \mu}{\sigma}, y_{min} = \frac{\text{lower_range} - \mu}{\sigma}`. 
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters 
    ----------
    x
        The input data
    mu 
        The location of the step
    sigma 
        The "width" of the step, because we are using an error function to define it
    hstep 
        The height of the step 
    lower_range
        The lower range on which to normalize the step PDF, default is to normalize from min to max x values
    upper_range 
        The upper range on which to normalize the step PDF

    """
    integral = nb_step_int(np.array([lower_range, upper_range]), mu, sigma, hstep)
    norm = integral[1]-integral[0]

    integrated_pdf = nb_step_int(x, mu, sigma, hstep)
    if norm == 0 :
        cdf = np.inf * integrated_pdf
    else:
        cdf = (integrated_pdf - integral[0])/norm # compute (cdf(x)-cdf(x_min))/norm
    return cdf


@nb.njit(**kwd)
def nb_step_scaled_pdf(x: np.ndarray, mu: float, sigma: float, hstep: float, lower_range: float, upper_range: float, area: float) -> np.ndarray:
    r"""
    Scaled step function pdf w/args mu, sigma, hstep, lower_range, upper_range, area.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters 
    ----------
    x
        The input data
    mu 
        The location of the step
    sigma 
        The "width" of the step, because we are using an error function to define it
    hstep 
        The height of the step 
    lower_range
        The lower range on which to normalize the step PDF, default is to normalize from min to max x values
    upper_range 
        The upper range on which to normalize the step PDF
    area
        The number to scale the distribution by

    """

    return area * nb_step_pdf(x,  mu, sigma, hstep, lower_range , upper_range)


@nb.njit(**kwd)
def nb_step_scaled_cdf(x: np.ndarray, mu: float, sigma: float, hstep: float, lower_range: float, upper_range: float, area: float) -> np.ndarray:
    r"""
    Scaled step function CDF w/args mu, sigma, hstep, lower_range, upper_range, area. Used for extended binned fits.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters 
    ----------
    x
        The input data
    mu 
        The location of the step
    sigma 
        The "width" of the step, because we are using an error function to define it
    hstep 
        The height of the step 
    lower_range
        The lower range on which to normalize the step PDF, default is to normalize from min to max x values
    upper_range 
        The upper range on which to normalize the step PDF
    area
        The number to scale the distribution by
    
    """
    
    return area * nb_step_cdf(x, mu, sigma, hstep, lower_range, upper_range)


class step_gen(pygama_continuous):

    def _argcheck(self, lower_range, upper_range, hstep, mu, sigma):
        return (upper_range>lower_range)
        
    def __init__(self, *args, **kwargs):
        self.x_lo = None
        self.x_hi = None
        super().__init__(self)

    def _pdf(self, x: np.ndarray, lower_range: float, upper_range: float, hstep: float, mu, sigma) -> np.ndarray:
        x.flags.writeable = True
        return nb_step_pdf(x, mu[0], sigma[0], hstep[0], lower_range[0], upper_range[0])
    def _cdf(self, x: np.ndarray, lower_range: float, upper_range: float, hstep: float, mu, sigma) -> np.ndarray:
        x.flags.writeable = True
        return nb_step_cdf(x, mu[0], sigma[0], hstep[0], lower_range[0], upper_range[0])

    def get_pdf(self, x: np.ndarray, lower_range: float, upper_range: float, hstep: float, mu: float, sigma: float) -> np.ndarray:
        return nb_step_pdf(x, mu, sigma, hstep, lower_range, upper_range)
    def get_cdf(self, x: np.ndarray, lower_range: float, upper_range: float,  hstep: float, mu: float, sigma: float) -> np.ndarray:
        return nb_step_cdf(x, mu, sigma, hstep, lower_range, upper_range)

    # Because step is only defined on a user specified range, we don't need to return a different pdf_norm, just alias get_pdf and get_cdf
    def pdf_norm(self, x: np.ndarray, x_lower: float, x_upper: float, hstep: float, mu: float, sigma: float) -> np.ndarray: 
        return nb_step_pdf(x, mu, sigma, hstep, x_lower, x_upper)
    def cdf_norm(self, x: np.ndarray, x_lower: float, x_upper: float, hstep: float, mu: float, sigma: float) -> np.ndarray: 
        return nb_step_cdf(x, mu, sigma, hstep, x_lower, x_upper)

    def pdf_ext(self, x: np.ndarray, area: float, hstep: float, lower_range: float, upper_range: float, mu: float, sigma: float) -> np.ndarray:
        return nb_step_scaled_cdf(np.array([self.x_lo, self.x_hi]), mu, sigma, hstep, lower_range, upper_range, area)[1]-nb_step_scaled_cdf(np.array([self.x_lo, self.x_hi]), mu, sigma, hstep, lower_range, upper_range, area)[0], nb_step_scaled_pdf(x, mu, sigma, hstep, lower_range, upper_range, area)
    def cdf_ext(self, x: np.ndarray, area: float, hstep: float, lower_range: float, upper_range: float, mu: float, sigma: float) -> np.ndarray:
        return nb_step_scaled_cdf(x, mu, sigma, hstep, lower_range, upper_range, area)

    def required_args(self) -> tuple[str, str, str, str, str]:
        return  "lower_range", "upper_range", "hstep", "mu", "sigma"

step = step_gen(name='step')