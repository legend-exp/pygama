"""
Double Gaussian distributions with different backgrounds for pygama
"""
import math

import numpy as np

from pygama.math._distributions.gauss import gauss_norm
from pygama.math._distributions.step import step_pdf


def Am_double(x,  n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3, n_bkg1, hstep1, n_bkg2, hstep2,
             lower_range=np.inf , upper_range=np.inf, components=False):
    """
    A Fit function exclusevly for a 241Am 99keV and 103keV lines situation
    Consists of

     - three gaussian peaks (two lines + one bkg line in between)
     - two steps (for the two lines)
     - two tails (for the two lines)
    """
    bkg1 = n_bkg1*step_pdf(x, mu1, sigma1, hstep1, lower_range, upper_range )
    bkg2 = n_bkg2*step_pdf(x, mu2, sigma2, hstep2, lower_range, upper_range)
    if np.any(bkg1<0) or np.any(bkg2<0):
        return 0, np.zeros_like(x)
    sig1 = n_sig1*gauss_norm(x,mu1,sigma1)
    sig2 = n_sig2* gauss_norm(x,mu2,sigma2)
    sig3 = n_sig3* gauss_norm(x,mu3,sigma3)
    if components ==False:
        return sig1+sig2+sig3+bkg1+bkg2
    else:
        return sig1,sig2,sig3,bkg1,bkg2


def extended_Am_double(x,  n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3,
                       n_bkg1, hstep1, n_bkg2, hstep2,
                     lower_range=np.inf , upper_range=np.inf, components=False):
    """
    A Fit function exclusevly for a 241Am 99keV and 103keV lines situation
    Consists of

     - three extended gaussian peaks (two lines + one bkg line in between)
     - two steps (for the two lines)
     - two tails (for the two lines)
    """
    if components ==False:
        return n_sig1+n_sig2+n_sig3 + n_bkg1+n_bkg2, Am_double(n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2,
                                                               n_sig3, mu3,sigma3,
                                                               n_bkg1, hstep1, n_bkg2, hstep2,
                                                                 lower_range, upper_range)
    else:
        sig1,sig2,sig3,bkg1,bkg2 = Am_double(n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3,
                                             n_bkg1, hstep1, n_bkg2, hstep2,
                                             lower_range , upper_range,components=components)
        return n_sig1+n_sig2+n_sig3 + n_bkg1+n_bkg2, sig1,sig2,sig3,bkg1,bkg2


def double_gauss_pdf(x,  n_sig1,  mu1, sigma1, n_sig2, mu2,sigma2,n_bkg,hstep,
                     lower_range=np.inf, upper_range=np.inf, components=False):
    """
    A Fit function exclusevly for a 133Ba 81keV peak situation
    Consists of

     - two gaussian peaks (two lines)
     - one step
    """
    bkg = n_bkg*step_pdf(x, mu1, sigma1, hstep, lower_range, upper_range)
    if np.any(bkg<0):
        return 0, np.zeros_like(x)
    sig1 = n_sig1*gauss_norm(x,mu1,sigma1)
    sig2 = n_sig2* gauss_norm(x,mu2,sigma2)
    if components == False:
        return sig1 + sig2 + bkg
    else:
        return sig1, sig2, bkg


def extended_double_gauss_pdf(x,  n_sig1,  mu1, sigma1, n_sig2, mu2,sigma2,n_bkg,hstep,
                     lower_range=np.inf , upper_range=np.inf, components=False):
    """
    A Fit function exclusevly for a 133Ba 81keV peak situation
    Consists of

     - two gaussian peaks (two lines)
     - one step
    """
    if components == False:
        pdf = double_gauss_pdf(x,  n_sig1,  mu1, sigma1, n_sig2, mu2,sigma2,n_bkg,hstep,
                     lower_range, upper_range)
        return n_sig1+n_sig2+n_bkg, pdf
    else:
        sig1, sig2, bkg = double_gauss_pdf(x,  n_sig1,  mu1, sigma1, n_sig2, mu2,sigma2,n_bkg,hstep,
                     lower_range, upper_range,components=components)
        return n_sig1+n_sig2+n_bkg, sig1, sig2, bkg
