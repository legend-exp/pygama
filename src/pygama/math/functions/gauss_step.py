"""
Gaussian distributions with a step function for pygama
"""
import sys

import numba as nb
import numpy as np

from pygama.math.functions.gauss import nb_gauss_cdf, nb_gauss_norm
from pygama.math.functions.step import nb_step_cdf, nb_step_pdf

kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_gauss_step_pdf(x, n_sig, mu, sigma, n_bkg, hstep, lower_range=np.inf, upper_range=np.inf, components=False):
    """
    PDF for Gaussian on top of a step background
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        Input data
    n_sig : float
        Number of counts in the signal
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    n_bkg : float
        The number of counts in the background
    hstep : float
        The height of the step function background
    lower_range : float
        Lower bound of the step function
    upper_range : float
        Upper bound of the step function
    components : bool
        If true, returns the signal and background components separately
    """
    try:
        bkg= nb_step_pdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
            bkg= np.zeros_like(x, dtype=np.float64)
    pdf = n_sig*nb_gauss_norm(x,mu,sigma) +\
          n_bkg*bkg
    if components ==False:
        return pdf
    else:
        return n_sig*nb_gauss_norm(x,mu,sigma), n_bkg*bkg


@nb.njit(**kwd)
def nb_gauss_step_cdf(x, n_sig, mu, sigma, n_bkg, hstep, lower_range=np.inf, upper_range=np.inf, components=False):
    """
    CDF for Gaussian on step background
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        Input data
    n_sig : float
        Number of counts in the signal
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    n_bkg : float
        The number of counts in the background
    hstep : float
        The height of the step function background
    lower_range : float
        Lower bound of the step function
    upper_range : float
        Upper bound of the step function
    components : bool
        If true, returns the signal and background components separately
    """
    try:
        bkg = nb_step_cdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
        bkg= np.zeros_like(x, dtype=np.float64)
    if components ==False:
        cdf = (1/(n_sig+n_bkg))*(n_sig*nb_gauss_cdf(x, mu, sigma) +\
          n_bkg*bkg)
        return cdf
    else:
        return (1/(n_sig+n_bkg))*n_sig*nb_gauss_cdf(x, mu, sigma), (1/(n_sig+n_bkg))*(n_bkg*bkg)


@nb.njit(**kwd)
def nb_gauss_step(x, n_sig, mu, sigma, n_bkg, hstep, lower_range=np.inf, upper_range=np.inf, components=False):
    """
    Pdf for Gaussian on step background for Compton spectrum, returns also the total number of events for extended unbinned fits
    As a Numba JIT function, it runs slightly faster than
    'out of the box' functions.

    Parameters
    ----------
    x : array-like
        Input data
    n_sig : float
        Number of counts in the signal
    mu : float
        The centroid of the Gaussian
    sigma : float
        The standard deviation of the Gaussian
    n_bkg : float
        The number of counts in the background
    hstep : float
        The height of the step function background
    lower_range : float
        Lower bound of the step function
    upper_range : float
        Upper bound of the step function
    components : bool
        If true, returns the signal and background components separately
    """
    if components ==False:
        return n_sig+n_bkg , nb_gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep, lower_range, upper_range)
    else:
        sig, bkg = nb_gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep,lower_range, upper_range, components=True)
        return n_sig+n_bkg, sig, bkg
