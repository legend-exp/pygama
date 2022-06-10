"""
Convenience functions to select distributions
"""
import numpy as np

from pygama.math.binned_fitting import hpge_peak_fwhm
from pygama.math.distributions import (
    gauss_step,
    extended_hpge_peak_pdf,
    gauss_step_cdf,
    gauss_step_pdf,
    hpge_peak_cdf,
    hpge_peak_pdf,
)


def get_mu_func(func, pars, cov = None, errors=None):
    """
    Function to select which distribution to get the centroid from
    """
    if  func == gauss_step_cdf or func == gauss_step_pdf or func == gauss_step:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return mu, errors[1]
        elif cov is not None:
            return mu, np.sqrt(cov[1][1])
        else:
            return mu

    elif  func == hpge_peak_cdf or func == hpge_peak_pdf or func == extended_hpge_peak_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return mu, errors[1]
        elif cov is not None:
            return mu, np.sqrt(cov[1][1])
        else:
            return mu

    else:
        print(f'get_mu_func not implemented for {func.__name__}')
        return None


def get_fwhm_func(func, pars, cov = None):
    """
    Function to select which distribution to get the FWHM of
    """
    if  func == gauss_step_cdf or func == gauss_step_pdf or func == gauss_step:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if cov is None:
            return sigma*2*np.sqrt(2*np.log(2))
        else:
            return sigma*2*np.sqrt(2*np.log(2)), np.sqrt(cov[2][2])*2*np.sqrt(2*np.log(2))

    elif  func == hpge_peak_cdf or func == hpge_peak_pdf or func == extended_hpge_peak_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars

        return hpge_peak_fwhm(sigma, htail, tau, cov)
    else:
        print(f'get_fwhm_func not implemented for {func.__name__}')
        return None


def get_total_events_func(func, pars, cov = None, errors=None):
    """
    Function to select which distribution to get the total number of events from
    """
    if  func == gauss_step_cdf or func == gauss_step_pdf or func == gauss_step:
        if len(pars) ==5:
            n_sig, mu, sigma, n_bkg, hstep = pars
        elif len(pars) ==7:
            n_sig, mu, sigma, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return n_sig+n_bkg, np.sqrt(errors[0]**2 + errors[3]**2)
        elif cov is not None:
            return n_sig+n_bkg, np.sqrt(cov[0][0]**2 + cov[3][3]**2)
        else:
            return n_sig+n_bkg

    elif  func == hpge_peak_cdf or func == hpge_peak_pdf or func == extended_hpge_peak_pdf:
        if len(pars) ==7:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep = pars
        elif len(pars) ==9:
            n_sig, mu, sigma, htail, tau, n_bkg, hstep, low_range, high_range = pars
        if errors is not None:
            return n_sig+n_bkg, np.sqrt(errors[0]**2 + errors[5]**2)
        elif cov is not None:
            return n_sig+n_bkg, np.sqrt(cov[0][0]**2 + cov[5][5]**2)
        else:
            return n_sig+n_bkg
    else:
        print(f'get_total_events_func not implemented for {func.__name__}')
        return None
