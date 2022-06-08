"""
Gaussian distributions with tails for pygama
"""
import sys

import numba as nb
import numpy as np

from pygama.math._distributions.gauss import gauss_cdf, gauss_norm
from pygama.math.functions import gauss_tail_exact, nb_erf

limit = np.log(sys.float_info.max)/10
kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def gauss_tail_pdf(x, mu, sigma, tau):
    """
    A gaussian tail function template
    Can be used as a component of other fit functions w/args mu,sigma,tau
    """
    x = np.asarray(x)
    tmp = ((x-mu)/tau) + ((sigma**2)/(2*tau**2))
    tail_f = np.where(tmp < limit,
                      gauss_tail_exact(x, mu, sigma, tau),
                      gauss_tail_approx(x, mu, sigma, tau))
    return tail_f


@nb.njit(**kwd)
def gauss_tail_cdf(x,mu,sigma,tau, lower_range=np.inf , upper_range=np.inf):
    """
    CDF for gaussian tail
    """
    cdf = gauss_tail_integral(x,mu,sigma,tau)
    if lower_range ==np.inf and upper_range ==np.inf:
        integral = gauss_tail_integral(np.array([np.nanmin(x), np.nanmax(x)]), mu, sigma, tau)
    else:
        integral = gauss_tail_integral(np.array([lower_range, upper_range]), mu, sigma, tau)
    norm = integral[1]-integral[0]
    cdf =  (1/norm) * cdf
    c = 1-cdf[-1]
    return cdf+c


def gauss_with_tail_pdf(x, mu, sigma,  htail,tau, components=False):
    """
    Pdf for gaussian with tail
    """
    if htail < 0 or htail > 1:
        if components ==False:
            return np.full_like(x, np.nan, dtype='float64')
        else:
            return np.full_like(x, np.nan, dtype='float64'), np.full_like(x, np.nan, dtype='float64')

    peak = gauss_norm(x,mu,sigma)
    try:
        tail = gauss_tail_pdf(x, mu, sigma, tau)
    except ZeroDivisionError:
        tail = np.zeros_like(x, dtype=np.float64)
    if components ==False:
        return (1-htail)*peak + htail*tail
    else:
        return (1-htail)*peak, htail*tail


def gauss_with_tail_cdf(x, mu, sigma, htail,  tau, components=False):
    """
    Cdf for gaussian with tail
    """
    if htail < 0 or htail > 1:
        if components ==False:
            return np.full_like(x, np.nan, dtype='float64')
        else:
            return np.full_like(x, np.nan, dtype='float64'), np.full_like(x, np.nan, dtype='float64')

    peak = gauss_cdf(x,mu,sigma)
    try:
        tail = gauss_tail_cdf(x, mu, sigma, tau)
    except  ZeroDivisionError:
        tail = np.zeros_like(x, dtype=np.float64)
    if components==False:
        return (1-htail)*peak + htail*tail
    else:
        return (1-htail)*peak, htail*tail


@nb.njit(**kwd)
def gauss_tail_approx(x, mu, sigma, tau):
    """
    Approximate form of gaussian tail
    """
    den = 1/(sigma + tau*(x-mu)/sigma)
    tail_f = sigma * gauss_norm(x, mu, sigma) * den * (1.-tau*tau*den*den)
    return tail_f


@nb.njit(**kwd)
def gauss_tail_integral(x,mu,sigma,tau):
    """
    Integral for gaussian tail
    """
    abstau = np.abs(tau)
    part1 = (tau/(2*abstau)) * nb_erf((tau*(x-mu) )/(np.sqrt(2)*sigma*abstau))
    part2 =    tau * gauss_tail_pdf(x,mu,sigma,tau)
    return part1+part2


@nb.njit(**kwd)
def gauss_tail_norm(x,mu,sigma,tau, lower_range=np.inf , upper_range=np.inf):
    """
    Normalised gauss tail. Note: this is only needed when the fitting range
    does not include the whole tail
    """
    tail = gauss_tail_pdf(x,mu,sigma,tau)
    if lower_range ==np.inf and upper_range ==np.inf:
        integral = gauss_tail_integral(np.array([np.nanmin(x), np.nanmax(x)]), mu, sigma, tau)
    else:
        integral = gauss_tail_integral(np.array([lower_range, upper_range]), mu, sigma, tau)
    norm = integral[1]-integral[0]
    return tail/norm
