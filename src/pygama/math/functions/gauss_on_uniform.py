r"""
Provide a convenience function for a Gaussian on top of a uniform distribution. 

Parameters 
----------
n_sig 
    The area of the Gaussian distribution
mu, sigma
    The location and scale of the first Gaussian
n_bkg 
    The area of the uniform distributions respectively
x_lower, x_upper
    The upper and lower bounds on which the uniform distribution is defined

Returns 
-------
gauss_on_uniform
    An instance of sum_dists and rv_continuous, has methods of pdf, cdf, etc.
"""
from pygama.math.functions.sum_dists import sum_dists

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.uniform import uniform

(n_sig, mu, sigma, n_bkg, x_lower, x_upper) = range(6) # the sum_dist array we will pass will be n_sig, mu, sigma, n_bkg, m, b
par_array = [(gaussian, [mu, sigma]), (uniform, [x_lower, x_upper])] 

gauss_on_uniform = sum_dists(par_array, [n_sig, n_bkg], "areas", parameter_names = ["n_sig", "mu", "sigma", "n_bkg", "x_lower", "x_upper"])
