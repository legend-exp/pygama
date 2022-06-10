"""
Step distributions for pygama
"""
import sys

import numba as nb
import numpy as np

from pygama.math.functions.error_function import nb_erf
from pygama.math.functions.gauss import gauss

kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def unnorm_step_pdf(x,  mu, sigma, hstep):
    """
    Unnormalised step function for use in pdfs
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.
    """
    invs = (np.sqrt(2)*sigma)
    z = (x-mu)/invs
    step_f = 1 + hstep * nb_erf(z)
    return step_f


@nb.njit(**kwd)
def step_pdf(x,  mu, sigma, hstep, lower_range=np.inf , upper_range=np.inf):
    """
    Normalised step function w/args mu, sigma, hstep
    Can be used as a component of other fit functions
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.
    """
    step_f = unnorm_step_pdf(x,  mu, sigma, hstep)
    if lower_range ==np.inf and upper_range ==np.inf:
        integral = step_int(np.array([np.nanmin(x), np.nanmax(x)]), mu, sigma, hstep)
    else:
        integral = step_int(np.array([lower_range, upper_range]), mu, sigma, hstep)

    norm = integral[1]-integral[0]
    return step_f/norm


@nb.njit(**kwd)
def step_cdf(x,mu,sigma, hstep, lower_range=np.inf , upper_range=np.inf):
    """
    CDF for step function w/args mu, sigma, hstep
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.
    """
    cdf = step_int(x,mu,sigma,hstep)
    if lower_range ==np.inf and upper_range ==np.inf:
        integral = step_int(np.array([np.nanmin(x), np.nanmax(x)]), mu, sigma, hstep)
    else:
        integral = step_int(np.array([lower_range, upper_range]), mu, sigma, hstep)
    norm = integral[1]-integral[0]
    cdf =  (1/norm) * cdf
    c = 1-cdf[-1]
    return cdf+c


@nb.njit(**kwd)
def step_int(x,mu,sigma, hstep):
    """
    Integral of step function w/args mu, sigma, hstep
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.
    """
    part1 = x+hstep*(x-mu)*nb_erf((x-mu)/(np.sqrt(2)*sigma))
    part2 = - np.sqrt(2/np.pi)*hstep*sigma*gauss(x,mu,sigma)
    return  part1-part2
