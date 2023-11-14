r"""
Provide a convenience function for a Gaussian on top of a step distribution. 

Parameters 
----------
area1
    The area of the Gaussian distribution
mu, sigma
    The location and scale of the first Gaussian, and step function
area2 
    The area of the step function
hstep
    The height of the step function
lower_range, upper_range 
    The lower and upper bounds of the support of the step function 

Returns 
-------
gauss_on_step
    An instance of sum_dists and rv_continuous, has methods of pdf, cdf, etc.
    
Notes 
-----
The step function shares a mu and sigma with the step function 
The parameter array must have ordering (area1, mu, sigma, area2, hstep, lower_range, upper_range) 
"""
from pygama.math.functions.sum_dists import sum_dists

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.step import step


(n_sig, mu, sigma, n_bkg, hstep, lower_range, upper_range) = range(7) # the sum_dist array we will pass will be n_sig, mu, sigma, n_bkg, m, b
par_array = [(gaussian, [mu, sigma]), (step, [lower_range, upper_range, hstep, mu, sigma])] 

gauss_on_step = sum_dists(par_array, [n_sig, n_bkg], "areas", parameter_names = ["n_sig", "mu", "sigma", "n_bkg", "hstep", "lower_range", "upper_range"])