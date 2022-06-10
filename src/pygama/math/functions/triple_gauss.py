"""
Double Gaussian distributions with different backgrounds for pygama
"""
import math

import numpy as np

from pygama.math.functions.gauss import gauss_norm
from pygama.math.functions.step import step_pdf


def triple_gauss_double_step_pdf(x,  n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3, n_bkg1, hstep1, n_bkg2, hstep2,
             lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Consists of

     - three gaussian peaks (two lines + one bkg line in between)
     - two steps (for the two lines)
     - two tails (for the two lines)
    """
    bkg1 = n_bkg1*step_pdf(x, mu1, sigma1, hstep1, lower_range, upper_range)
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


def triple_gauss_double_step(x,  n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3,
                       n_bkg1, hstep1, n_bkg2, hstep2,
                     lower_range=np.inf , upper_range=np.inf, components=False):
    """
    Consists of

     - three extended gaussian peaks (two lines + one bkg line in between)
     - two steps (for the two lines)
     - two tails (for the two lines)
    """
    if components ==False:
        return n_sig1+n_sig2+n_sig3 + n_bkg1+n_bkg2, triple_gauss_double_step_pdf(n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2,
                                                               n_sig3, mu3,sigma3,
                                                               n_bkg1, hstep1, n_bkg2, hstep2,
                                                                 lower_range, upper_range)
    else:
        sig1,sig2,sig3,bkg1,bkg2 = triple_gauss_double_step_pdf(n_sig1, mu1, sigma1,  n_sig2, mu2,sigma2, n_sig3, mu3,sigma3,
                                             n_bkg1, hstep1, n_bkg2, hstep2,
                                             lower_range , upper_range,components=components)
        return n_sig1+n_sig2+n_sig3 + n_bkg1+n_bkg2, sig1,sig2,sig3,bkg1,bkg2