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
                lower_range=np.inf , upper_range=np.inf,  components=False):
    """
    David Radford's HPGe peak shape PDF consists of a gaussian with tail signal
    on a step background
    """
    try:
        bkg= nb_step_pdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
        bkg = np.zeros_like(x, dtype=np.float64)
    if components ==False:
        sig = nb_gauss_with_tail_pdf(x, mu, sigma, htail,  tau)
        pdf = (n_bkg * bkg +\
             n_sig *  sig)
        return pdf
    else:
        peak, tail = nb_gauss_with_tail_pdf(x, mu, sigma, htail,  tau, components=components)
        return n_sig *peak, n_sig*tail, n_bkg * bkg


@nb.njit(**kwd)
def nb_hpge_peak_cdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep, lower_range=np.inf , upper_range=np.inf,  components=False):
    """
    Cdf for gaussian with tail signal and step background
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
                         lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Pdf for gaussian with tail signal and step background, also returns number of events
    """
    if components ==False:
        return n_sig + n_bkg, nb_hpge_peak_pdf(x, n_sig,  mu, sigma, htail, tau, n_bkg, hstep, lower_range, upper_range)
    else:
        peak, tail, bkg = nb_hpge_peak_pdf(x, n_sig,  mu, sigma, htail, tau, n_bkg, hstep,
                                      lower_range, upper_range,components=components)
        return n_sig + n_bkg, peak, tail, bkg
