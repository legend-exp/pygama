import sys

import numba as nb
import numpy as np

from scipy.stats import rv_continuous
from pygama.math.functions.sum_dists import sum_dists

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.exgauss import exgauss
from pygama.math.functions.step import step

from pygama.math.hpge_peak_fitting import hpge_peak_fwhm


class hpge_peak_gen(sum_dists):
    r"""
    Provide a convenience function for the HPGe peak shape. 

    A HPGe peak consists of a Gaussian
    on an Exgauss on a step function. 

    .. math::
    
        PDF = n_sig*((1-htail)*gauss + htail*exgauss) + n_bkg*step


    Called with 

    hpge_peak.get_pdf(x, params=[mu, sigma, tau, htail, n_sig, hstep, lower_range, upper_range, n_bkg])

    Parameters
    ----------
    x
        Input data
    mu
        The centroid of the Gaussian
    sigma
        The standard deviation of the Gaussian
    tau
        The characteristic scale of the Gaussian tail
    htail
        The height of the Gaussian tail
    n_sig
        The area of the gauss on exgauss
    hstep
        The height of the step function background
    lower_range
        Lower bound of the step function
    upper_range
        Upper bound of the step function
    n_bkg
        The area of the step background

    Returns 
    -------
    hpge_peak
        A subclass of sum_dists and rv_continuous, has methods of pdf, cdf, etc.

    Notes 
    ----- 
    The extended Gaussian distribution and the step distribution share the mu, sigma with the Gaussian
    """
    
    def __init__(self):
        
        (mu, sigma, tau, frac1, n_sig, hstep, lower_range, upper_range, n_bkg) = range(9)
        args = [gaussian, [mu, sigma, n_sig, frac1], exgauss, [tau, mu, sigma, n_sig, frac1], step, [hstep, lower_range, upper_range, mu, sigma, n_bkg, frac1, frac1] ]
        
        # throw the step distribution htail because we will override its frac and total area to 1
        
        super().__init__(*args, frac_flag = "both")
        
    def _link_pars(self, shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area):
        shape_pars, cum_len, areas, fracs, total_area = super()._link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
    
        fracs[0] = 1-fracs[1] # create (1-htail) for the gaussian, and htail for the exgauss
        fracs[2] = 1 # make sure that the step function has a frac of 1
        total_area[0] = 1 # make sure that the total area is also just 1
        
        
        return shape_pars, cum_len, areas, fracs, total_area 
    
    def get_req_args(self) -> tuple[str,str,str,str,str,str,str,str,str]:
        r"""
        Return the required args for this instance
        """
        return "mu", "sigma", "tau", "htail", "n_sig", "hstep", "lower_range", "upper_range", "n_bkg"
        

    def get_fwhm(self, pars: np.ndarray, cov: np.ndarray = None) -> tuple:
        r"""
        Get the fwhm value from the output of a fit quickly
        Need to overload this to use hpge_peak_fwhm (to avoid a circular import) for when self is an hpge peak, 
        and otherwise returns 2sqrt(2log(2))*sigma

        Parameters 
        ----------
        pars 
            Array of fit parameters
        cov 
            Optional, array of covariances for calculating error on the fwhm


        Returns 
        -------
        fwhm, error 
            the value of the fwhm and its error
        """

        req_args = np.array(self.get_req_args())
        sigma_idx = np.where(req_args == "sigma")[0][0]

        if ("htail" in req_args) and ("hstep" in req_args): #having both the htail and hstep means it is an exgauss on a step
            htail_idx = np.where(req_args == "htail")[0][0]
            tau_idx = np.where(req_args == "tau")[0][0]
            
            # We need to move around the cov matrix because hpge_peak_fwhm has a different ordering on the cov matrix...
            mu_idx = np.where(req_args == "mu")[0][0]
            hstep_idx = np.where(req_args == "hstep")[0][0]
            n_bkg_idx = np.where(req_args == "n_bkg")[0][0]
            n_sig_idx = np.where(req_args == "n_sig")[0][0]
            rearranged_cov = np.diag(np.array([cov[mu_idx][mu_idx], cov[sigma_idx][sigma_idx], cov[hstep_idx][hstep_idx], cov[htail_idx][htail_idx], cov[tau_idx][tau_idx], cov[n_bkg_idx][n_bkg_idx], cov[n_sig_idx][n_sig_idx] ]))

            return hpge_peak_fwhm(pars[sigma_idx], pars[htail_idx], pars[tau_idx], rearranged_cov)

        else: 
            if cov is None:
                return pars[sigma_idx]*2*np.sqrt(2*np.log(2))
            else:
                return pars[sigma_idx]*2*np.sqrt(2*np.log(2)), np.sqrt(cov[sigma_idx][sigma_idx])*2*np.sqrt(2*np.log(2))


    
hpge_peak = hpge_peak_gen()