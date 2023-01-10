import sys

import numba as nb
import numpy as np

from scipy.stats import rv_continuous
from pygama.math.functions.sum_dists import sum_dists


from pygama.math.functions.gauss import gaussian
from pygama.math.functions.exgauss import exgauss


class gauss_on_exgauss_gen(sum_dists):
    r"""
    Provide a convenience function for a Gaussian on top of an extended Gaussian. The correct usage is :func:`gauss_on_exgauss.pdf(x, pars = [mu, sigma, htail, tau])`

    Parameters
    ----------
    mu
        The location and scale of the first Gaussian
    htail
        The height of the tail
    tau
        The characteristic scale of the extended Gaussian tail

    Returns
    -------
    gauss_on_exgauss
        A subclass of :class:`sum_dists` and :class:`rv_continuous`, has methods of :func:`pdf`, :func:`cdf`, etc.

    Notes
    -----
    The extended Gaussian distribution shares the mu, sigma with the Gaussian
    """

    def __init__(self):
        
        (mu, sigma, frac1, tau) = range(4)
        args = [gaussian, [mu, sigma, frac1], exgauss, [tau, mu, sigma, frac1, frac1]] # pass it :param:`frac1` for the total area because we don't care about it at all
        
        
        super().__init__(*args, frac_flag = "fracs")
        
    def _link_pars(self, shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area):
        shape_pars, cum_len, areas, fracs, total_area = super()._link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
    
        fracs[0] = 1-fracs[1] # create :math:`(1-htail)` for the gaussian, and :math:`htail` for the exgauss
        total_area[0] = 1 # just overwrite the total area because we don't want to fit it
        
        
        return shape_pars, cum_len, areas, fracs, total_area 

    def get_req_args(self) -> tuple[str, str, str, str] : 
        r"""
        Return the required args for this instance 
        """
        return "mu", "sigma", "htail", "tau"

gauss_on_exgauss = gauss_on_exgauss_gen()