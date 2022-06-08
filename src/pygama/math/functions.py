"""
pygama convenience functions
"""
import math
import sys

import numba as nb
import numpy as np
from scipy.stats import crystalball

limit = np.log(sys.float_info.max)/10
kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_erf(x):
    """
    Numba version of error function
    """
    y = np.empty_like(x)
    for i in nb.prange(len(x)):
        y[i] = math.erf(x[i])
    return y


@nb.njit(**kwd)
def nb_erfc(x):
    """
    Numba version of complementary error function
    """
    y = np.empty_like(x)
    for i in nb.prange(len(x)):
        y[i] = math.erfc(x[i])
    return y


@nb.njit(**kwd)
def gauss(x, mu, sigma):
    """
    Gaussian, unnormalised for use in building pdfs, w/ args: mu, sigma.
    """
    if sigma ==0: invs=np.nan
    else: invs = 1.0 / sigma
    z = (x - mu) * invs
    return np.exp(-0.5 * z ** 2)


@nb.njit(**kwd)
def gauss_amp(x, mu, sigma, a):
    """
    Gaussian with height as a parameter for fwhm etc. args mu sigma, amplitude
    """
    return a * gauss(x,mu,sigma)


@nb.njit(**kwd)
def step_int(x,mu,sigma, hstep):
    """
    Integral of step function w/args mu, sigma, hstep
    """
    part1 = x+hstep*(x-mu)*nb_erf((x-mu)/(np.sqrt(2)*sigma))
    part2 = - np.sqrt(2/np.pi)*hstep*sigma*gauss(x,mu,sigma)
    return  part1-part2


@nb.njit(**kwd)
def gauss_tail_exact(x, mu, sigma, tau):
    """
    Exact form of gaussian tail
    """
    tmp = ((x-mu)/tau) + ((sigma**2)/(2*tau**2))
    abstau = np.absolute(tau)
    tmp = np.where(tmp < limit, tmp, limit)
    z = (x-mu)/sigma
    tail_f = (1/(2*abstau)) * np.exp(tmp) * nb_erfc( (tau*z + sigma)/(np.sqrt(2)*abstau))
    return tail_f


def xtalball(x, mu, sigma, A, beta, m):
    """
    power-law tail plus gaussian https://en.wikipedia.org/wiki/Crystal_Ball_function
    """
    return A * crystalball.pdf(x, beta, m, loc=mu, scale=sigma)


def poly(x, pars):
    """
    A polynomial function with pars following the polyfit convention
    """
    result = x*0 # do x*0 to keep shape of x (scalar or array)
    if len(pars) == 0: return result
    result += pars[-1]
    for i in range(1, len(pars)):
        result += pars[-i-1]*x
        x = x*x
    return result
