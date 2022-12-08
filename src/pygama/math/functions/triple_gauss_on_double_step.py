import sys

import numba as nb
import numpy as np

from scipy.stats import rv_continuous
from pygama.math.functions.sum_dists import sum_dists

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.step import step



class triple_gauss_on_double_step_gen(sum_dists):
    r"""
    Provide a convenience function for three gaussians on two steps

    Parameters 
    ----------
    mu_1, sigma_1 
        The location and scale of the first Gaussian
    mu_2, sigma_2
        The location and scale of the second Gaussian 
    mu_3, sigma_3 
        The location and scale of the third Gaussian 
    hstep_1 
        The height of the first step function
    hstep_2
        The height of the second step function 
    lower_range
        The lower range to compute the normalization of the step functions 
    upper_range
        The upper range to compute the normalization of the step functions 

    Example
    ------- 
    triple_gauss_on_double_step.get_pdf(x, pars = [n_sig1, mu1, sigma1, n_sig2, mu2, sigma2, n_sig3, mu3, sigma3, n_bkg1, hstep1, n_bkg2, hstep2, lower_range, upper_range])

    Returns 
    -------
    triple_gauss_on_double_step
        A subclass of sum_dists and rv_continuous, has methods of pdf, cdf, etc.

    Notes 
    ----- 
    The first step function shares the mu_1, sigma_1 with the first Gaussian,
    and the second step function shares the mu_2, sigma_2 with the second Gaussian
    """
    
    def __init__(self):
        
        (area1, mu1, sigma1, area2, mu2, sigma2, area3, mu3, sigma3, area4, hstep1, area5, hstep2, lower_range, upper_range) = range(15)
        args = [gaussian, [mu1, sigma1, area1], gaussian, [mu2, sigma2, area2], gaussian, [mu3, sigma3, area3], step, [hstep1, lower_range, upper_range, mu1, sigma1, area4], step, [hstep2, lower_range, upper_range, mu2, sigma2, area5]]
        
        super().__init__(*args, frac_flag = "areas")

    
    def get_req_args(self) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str, str, str, str]:
        r"""
        Return the required arguments for this instance
        """
        return "n_sig1", "mu1", "sigma1", "n_sig2", "mu2", "sigma2",  "n_sig3", "mu3", "sigma3", "n_bkg1", "hstep1", "n_bkg2", "hstep2", "lower_range", "upper_range"
        
triple_gauss_on_double_step = triple_gauss_on_double_step_gen()
