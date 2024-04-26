r"""
Provide a convenience function for a Gaussian on top of a step distribution.

Parameters
----------
x_lo, x_hi
    The lower and upper bounds of the support of the step function
area1
    The area of the Gaussian distribution
mu, sigma
    The location and scale of the first Gaussian, and step function
area2
    The area of the step function
hstep
    The height of the step function

Returns
-------
gauss_on_step
    An instance of SumDists and rv_continuous, has methods of pdf, cdf, etc.

Notes
-----
The step function shares a mu and sigma with the step function
The parameter array must have ordering (x_lo, x_hi, area1, mu, sigma, area2, hstep)
"""

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.step import step
from pygama.math.functions.sum_dists import SumDists

(x_lo, x_hi, n_sig, mu, sigma, n_bkg, hstep) = range(
    7
)  # the sum_dist array we will pass will be x_lo, x_hi, n_sig, mu, sigma, n_bkg, hstep
par_array = [(gaussian, [mu, sigma]), (step, [x_lo, x_hi, mu, sigma, hstep])]

gauss_on_step = SumDists(
    par_array,
    [n_sig, n_bkg],
    "areas",
    parameter_names=["x_lo", "x_hi", "n_sig", "mu", "sigma", "n_bkg", "hstep"],
    name="gauss_on_step",
)
