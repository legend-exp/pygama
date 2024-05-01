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
x_lo
    The lower bound on which to normalize the linear distribution
x_hi
    The upper bound on which to normalize the linear distribution


Returns
-------
gauss_on_linear
    An instance of SumDists and rv_continuous, has methods of pdf, cdf, etc.
"""

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.linear import linear
from pygama.math.functions.sum_dists import SumDists

(linear_x_lo, linear_x_hi, n_sig, mu, sigma, n_bkg, m, b) = range(
    8
)  # the sum_dist array we will pass will be x_lo, x_hi, n_sig, mu, sigma, n_bkg, m, b
par_array = [(gaussian, [mu, sigma]), (linear, [linear_x_lo, linear_x_hi, m, b])]

gauss_on_linear = SumDists(
    par_array,
    [n_sig, n_bkg],
    "areas",
    parameter_names=["x_lo", "x_hi", "n_sig", "mu", "sigma", "n_bkg", "m", "b"],
    name="gauss_on_linear",
)
