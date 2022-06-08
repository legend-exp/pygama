"""
Radford Ge peak shape distributions for pygama
"""
import numpy as np

from pygama.math._distributions.gauss_with_tail import gauss_with_tail_cdf, gauss_with_tail_pdf
from pygama.math._distributions.step import step_cdf, step_pdf


def radford_pdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep,
                lower_range=np.inf , upper_range=np.inf,  components=False):
    """
    David Radford's HPGe peak shape PDF consists of a gaussian with tail signal
    on a step background
    """
    try:
        bkg= step_pdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
        bkg = np.zeros_like(x, dtype=np.float64)
    if components ==False:
        sig = gauss_with_tail_pdf(x, mu, sigma, htail,  tau)
        pdf = (n_bkg * bkg +\
             n_sig *  sig)
        return pdf
    else:
        peak, tail = gauss_with_tail_pdf(x, mu, sigma, htail,  tau, components=components)
        return n_sig *peak, n_sig*tail, n_bkg * bkg


def radford_cdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep, lower_range=np.inf , upper_range=np.inf,  components=False):
    """
    Cdf for gaussian with tail signal and step background
    """
    try:
        bkg = step_cdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
        bkg= np.zeros_like(x, dtype=np.float64)
    if components ==False:
        sig = gauss_with_tail_cdf(x, mu, sigma, htail)
        pdf = (1/(n_sig+n_bkg))*(n_sig*gauss_with_tail_cdf(x, mu, sigma, htail,tau) +\
            n_bkg*bkg)
        return pdf
    else:
        peak, tail = gauss_with_tail_cdf(x, mu, sigma, htail, components= True)
        return (n_sig/(n_sig+n_bkg))*peak, (n_sig/(n_sig+n_bkg))*tail, (n_bkg/(n_sig+n_bkg))*bkg


def extended_radford_pdf(x, n_sig, mu, sigma, htail, tau, n_bkg, hstep,
                         lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Pdf for gaussian with tail signal and step background, also returns number of events
    """
    if components ==False:
        return n_sig + n_bkg, radford_pdf(x, n_sig,  mu, sigma, htail, tau, n_bkg, hstep, lower_range, upper_range)
    else:
        peak, tail, bkg = radford_pdf(x, n_sig,  mu, sigma, htail, tau, n_bkg, hstep,
                                      lower_range, upper_range,components=components)
        return n_sig + n_bkg, peak, tail, bkg
