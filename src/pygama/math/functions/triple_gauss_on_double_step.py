r"""
Provide a convenience function for three gaussians on two steps

Parameters 
----------
area1, mu_1, sigma_1 
    The area, location, and scale of the first Gaussian
area2, mu_2, sigma_2
    The area, location, and scale of the second Gaussian 
area3, mu_3, sigma_3 
    The area, location, and scale of the third Gaussian
area4, hstep_1 
    The area and height of the first step function
area5, hstep_2
    The area and height of the second step function 
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
import sys

import numba as nb
import numpy as np

from scipy.stats import rv_continuous
from pygama.math.functions.sum_dists import sum_dists

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.gauss_on_step import gauss_on_step

(area1, mu1, sigma1, area4, hstep1, area2, mu2, sigma2, area5, hstep2, lower_range, upper_range) = range(12)
par_array = [(gauss_on_step, [area1, mu1, sigma1, area4, hstep1, lower_range, upper_range]), (gauss_on_step, [area2, mu2, sigma2, area5, hstep2, lower_range, upper_range])] 

double_gauss_on_double_step = sum_dists(par_array, [], None, parameter_names = ["area1", "mu1", "sigma1", "area4", "hstep1", "area2", "mu2", "sigma2", "area5", "hstep2", "lower_range", "upper_range"])

(area1, mu1, sigma1, area2, mu2, sigma2, area3, mu3, sigma3, area4, hstep1, area5, hstep2, lower_range, upper_range) = range(15)

par_array = [(gaussian, [mu3, sigma3]), (double_gauss_on_double_step, [area1, mu1, sigma1, area4, hstep1, area2, mu2, sigma2, area5, hstep2, lower_range, upper_range])] 
triple_gauss_on_double_step = sum_dists(par_array, [area3], "one_area", parameter_names=["area1", "mu1", "sigma1", "area2", "mu2", "sigma2", "area3", "mu3", "sigma3", "area4", "hstep1", "area5", "hstep2", "lower_range", "upper_range"])