"""
Gaussian distributions for pygama
"""
import sys

import numba as nb
import numpy as np

from pygama.math.functions.error_function import nb_erf

kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_gauss(x, mu, sigma):
    """
    Gaussian, unnormalised for use in building pdfs
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        Input data
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    """
    if sigma ==0: invs=np.nan
    else: invs = 1.0 / sigma
    z = (x - mu) * invs
    return np.exp(-0.5 * z ** 2)


@nb.njit(**kwd)
def nb_gauss_amp(x, mu, sigma, a):
    """
    Gaussian with height as a parameter for fwhm etc.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        Input data
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    a : float
        The amplitude of the Gaussian
    """
    return a * nb_gauss(x,mu,sigma)


@nb.njit(**kwd)
def nb_gauss_norm(x, mu, sigma):
    """
    Normalised Gaussian, w/ args: mu, sigma.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        The input data
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    """
    if sigma ==0: invs=np.nan
    else: invs = 1.0 / sigma
    z = (x - mu) * invs
    invnorm = invs/ np.sqrt(2 * np.pi)
    return np.exp(-0.5 * z ** 2) * invnorm


@nb.njit(**kwd)
def nb_gauss_cdf(x, mu, sigma):
    """
    gaussian cdf, w/ args: mu, sigma.
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        The input data
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    """
    return 1/2 * (1 + nb_erf((x - mu)/(sigma*np.sqrt(2))))


@nb.njit(**kwd)
def nb_gauss_pdf(x, mu, sigma, n_sig):
    """
    Basic Gaussian pdf args; mu, sigma, n_sig (number of signal events)
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        The input data
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    n_sig : float
        The number of counts in the signal
    """
    return n_sig * nb_gauss_norm(x,mu,sigma)
