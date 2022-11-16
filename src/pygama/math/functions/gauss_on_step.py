import sys

import numba as nb
import numpy as np

from scipy.stats import rv_continuous
from pygama.math.functions.sum_dists import sum_dists

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.step import step

class gauss_on_step_gen(sum_dists):
    r"""
    Provide a convenience function for a Gaussian on top of a step distribution. 

    Parameters 
    ----------
    mu, sigma
        The location and scale of the first Gaussian, and step function
    area1
        The area of the Gaussian distribution
    hstep
        The height of the step function
    lower_range, upper_range 
        The lower and upper bounds of the support of the step function 
    area2 
        The area of the step function

    Returns 
    -------
    gauss_on_step
        A subclass of sum_dists and rv_continuous, has methods of pdf, cdf, etc.
        
    Notes 
    -----
    The step function shares a mu and sigma with the step function 
    """
    
    def __init__(self):
        
        (mu, sigma, area1, hstep, lower_range, upper_range, area2) = range(7)
        args = [gaussian, [mu, sigma, area1], step, [hstep, lower_range, upper_range, mu, sigma, area2]] 
        
        super().__init__(*args, frac_flag = "areas")

    # We need to redefine the get_cdf because in the previous pygama definition of the function, 
    # the fractions were normalized by the total sum 
    def _cdf(self, x, params):
        
        cdfs = self.dists 
        fracs = self.fracs 
        areas = self.areas 
        total_area = self.total_area
        
        shape_par_idx = self.shape_par_idx
        area_idx = self.area_idx
        frac_idx = self.frac_idx
        total_area_idx = self.total_area_idx
        
        # right now, params COULD contain areas and/or fracs... split it off
        
        shape_pars, cum_len, areas, fracs, total_area = self.link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
        


        probs = [cdfs[i].cdf(x, *shape_pars[cum_len[i]:cum_len[i+1]]) for i in range(len(cdfs))]
        
        prefactor = total_area*fracs*areas/np.sum(areas) # note the appearance of the sum of the areas in the denominator

        if self.components:
            probs = (probs.T*prefactor).T
            return probs
        else:
            probs = prefactor@probs
            return probs


    def get_cdf(self, x, params):
        cdfs = self.dists 
        fracs = self.fracs 
        areas = self.areas 
        total_area = self.total_area
        
        shape_par_idx = self.shape_par_idx
        area_idx = self.area_idx
        frac_idx = self.frac_idx
        total_area_idx = self.total_area_idx
        
        # right now, params COULD contain areas and/or fracs... split it off
        
        shape_pars, cum_len, areas, fracs, total_area = self.link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
        
        probs = [cdfs[i].get_cdf(x, *shape_pars[cum_len[i]:cum_len[i+1]]) for i in range(len(cdfs))]
        
        prefactor = total_area*fracs*areas/np.sum(areas) # note the appearance of the sum of the areas in the denominator

        if self.components:
            probs = (probs.T*prefactor).T
            return probs
        else:
            probs = prefactor@probs
            return probs

    def get_req_args(self) -> tuple[str, str, str, str, str, str, str]:  
        r""" 
        Return the required arguments
        """
        return "mu", "sigma", "n_sig", "hstep", "lower_range", "upper_range", "n_bkg"


gauss_on_step = gauss_on_step_gen()