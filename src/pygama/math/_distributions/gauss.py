"""
Gaussian distributions for pygama
"""
import sys

import numba as nb
import numpy as np

from pygama.math.functions import nb_erf

limit = np.log(sys.float_info.max)/10
kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def gauss_norm(x, mu, sigma):
    """
    Normalised Gaussian, w/ args: mu, sigma.
    """
    if sigma ==0: invs=np.nan
    else: invs = 1.0 / sigma
    z = (x - mu) * invs
    invnorm = invs/ np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z ** 2) * invnorm


@nb.njit(**kwd)
def gauss_cdf(x, mu, sigma):
    """
    gaussian cdf, w/ args: mu, sigma.
    """
    return 1/2 * (1 + nb_erf((x - mu)/(sigma*np.sqrt(2))))


@nb.njit(**kwd)
def gauss_pdf(x, mu, sigma, n_sig):
    """
    Basic Gaussian pdf args; mu, sigma, n_sig (number of signal events)
    """
    return n_sig * gauss_norm(x,mu,sigma)
