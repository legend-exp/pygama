"""
Gaussian distributions with a step function for pygama
"""
import sys

import numba as nb
import numpy as np


from pygama.math.distributions import gauss_norm, step_pdf, step_cdf, gauss_cdf


limit = np.log(sys.float_info.max)/10
kwd = {"parallel": False, "fastmath": True}


def gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep, lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Pdf for Gaussian on step background
    args: n_sig mu, sigma for the signal and n_bkg,hstep for the background
    """
    try:
        bkg= step_pdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
            bkg= np.zeros_like(x, dtype=np.float64)
    pdf = n_sig*gauss_norm(x,mu,sigma) +\
          n_bkg*bkg
    if components ==False:
        return pdf
    else:
        return n_sig*gauss_norm(x,mu,sigma), n_bkg*bkg


def gauss_step_cdf(x,  n_sig, mu, sigma,n_bkg, hstep, lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Cdf for Gaussian on step background
    args: n_sig mu, sigma for the signal and n_bkg,hstep for the background
    """
    try:
        bkg = step_cdf(x, mu, sigma, hstep, lower_range, upper_range)
    except ZeroDivisionError:
        bkg = np.zeros_like(x, dtype=np.float64)
    if np.any(bkg<0):
        bkg= np.zeros_like(x, dtype=np.float64)
    if components ==False:
        cdf = (1/(n_sig+n_bkg))*(n_sig*gauss_cdf(x, mu, sigma) +\
          n_bkg*bkg)
        return cdf
    else:
        return (1/(n_sig+n_bkg))*n_sig*gauss_cdf(x, mu, sigma), (1/(n_sig+n_bkg))*(n_bkg*bkg)


def extended_gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep, lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Pdf for Gaussian on step background for Compton spectrum, returns also the total number of events for extended unbinned fits
    args: n_sig mu, sigma for the signal and n_bkg, hstep for the background
    """
    if components ==False:
        return n_sig+n_bkg , gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep, lower_range, upper_range)
    else:
        sig, bkg = gauss_step_pdf(x,  n_sig, mu, sigma, n_bkg, hstep,lower_range, upper_range, components=True)
        return n_sig+n_bkg, sig, bkg