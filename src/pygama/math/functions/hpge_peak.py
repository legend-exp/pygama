"""
Radford Ge peak shape distributions for pygama
"""
import numba as nb
import numpy as np

from pygama.math.functions.gauss_with_tail import (
    nb_gauss_with_tail_cdf,
    nb_gauss_with_tail_pdf,
)
from pygama.math.functions.step import nb_step_cdf, nb_step_pdf

kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def nb_hpge_peak_pdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep,
                lower_range=np.inf, upper_range=np.inf, components=False):
    """
    HPGe peak shape PDF consists of a gaussian with tail signal
    on a step background.
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
    htail : float 
        The height of the Gaussian tail 
    tau : float 
        The characteristic scale of the Gaussian tail
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
        bkg = np.zeros_like(x, dtype=np.float64)
    if components == False:
        sig = nb_gauss_with_tail_pdf(x, mu, sigma, htail,  tau)
        pdf = (n_bkg * bkg +\
             n_sig *  sig)
        return pdf
    else:
        peak, tail = nb_gauss_with_tail_pdf(x, mu, sigma, htail,  tau, components=components)
        return n_sig *peak, n_sig*tail, n_bkg * bkg


@nb.njit(**kwd)
def nb_hpge_peak_cdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep, lower_range=np.inf, upper_range=np.inf, components=False):
    """
    Cdf for gaussian with tail signal and step background
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
    htail : float 
        The height of the Gaussian tail 
    tau : float 
        The characteristic scale of the Gaussian tail
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
        sig = nb_gauss_with_tail_cdf(x, mu, sigma, htail)
        pdf = (1/(n_sig+n_bkg))*(n_sig*nb_gauss_with_tail_cdf(x, mu, sigma, htail,tau) +\
            n_bkg*bkg)
        return pdf
    else:
        peak, tail = nb_gauss_with_tail_cdf(x, mu, sigma, htail, components= True)
        return (n_sig/(n_sig+n_bkg))*peak, (n_sig/(n_sig+n_bkg))*tail, (n_bkg/(n_sig+n_bkg))*bkg


@nb.njit(**kwd)
def nb_extended_hpge_peak_pdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep,
                         lower_range=np.inf, upper_range=np.inf, components=False):
    """
    PDF for gaussian with tail signal and step background, also returns number of events
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
    htail : float 
        The height of the Gaussian tail 
    tau : float 
        The characteristic scale of the Gaussian tail
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
        return n_sig + n_bkg, nb_hpge_peak_pdf(x, n_sig,  mu, sigma, htail, tau, n_bkg, hstep, lower_range, upper_range)
    else:
        peak, tail, bkg = nb_hpge_peak_pdf(x, n_sig,  mu, sigma, htail, tau, n_bkg, hstep,
                                      lower_range, upper_range,components=components)
        return n_sig + n_bkg, peak, tail, bkg
