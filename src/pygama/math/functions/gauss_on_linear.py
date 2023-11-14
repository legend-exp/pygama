import sys

r"""
Provide a convenience function for a Gaussian on top of a linear distribution. 

Parameters 
----------
area1 
    The area of the Gaussian distribution
mu, sigma
    The location and scale of the first Gaussian
area2 
    The area of the linear distributions respectively
m 
    The slope of the linear distribution 
b 
    The y intercept of the linear distribution
x_lower
    The lower bound on which to normalize the linear distribution
x_upper
    The upper bound on which to normalize the linear distribution


Returns 
-------
gauss_on_linear
    An instance of sum_dists and rv_continuous, has methods of pdf, cdf, etc.
"""
from pygama.math.functions.sum_dists import sum_dists

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.linear import linear

(n_sig, mu, sigma, n_bkg, m, b, linear_x_lo, linear_x_hi) = range(8) # the sum_dist array we will pass will be n_sig, mu, sigma, n_bkg, m, b
par_array = [(gaussian, [mu, sigma]), (linear, [linear_x_lo, linear_x_hi, m, b])] 

gauss_on_linear = sum_dists(par_array, [n_sig, n_bkg], "areas", parameter_names = ["n_sig", "mu", "sigma", "n_bkg", "m", "b", "x_lower", "x_upper"])
