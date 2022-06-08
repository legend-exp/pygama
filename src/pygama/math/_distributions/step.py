"""
Step distributions for pygama
"""
import sys

import numba as nb
import numpy as np

from pygama.math.functions import nb_erf, step_int

limit = np.log(sys.float_info.max)/10
kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def unnorm_step_pdf(x,  mu, sigma, hstep):
    """
    Unnormalised step function for use in pdfs
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
