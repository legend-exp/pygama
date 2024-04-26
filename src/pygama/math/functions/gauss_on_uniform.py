r"""
Provide a convenience function for a Gaussian on top of a uniform distribution.

Parameters
----------
x_lo, x_hi
    The upper and lower bounds on which the uniform distribution is defined
n_sig
    The area of the Gaussian distribution
mu, sigma
    The location and scale of the first Gaussian
n_bkg
    The area of the uniform distributions respectively

Returns
-------
gauss_on_uniform
    An instance of SumDists and rv_continuous, has methods of pdf, cdf, etc.
"""

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.sum_dists import SumDists
from pygama.math.functions.uniform import uniform

(x_lo, x_hi, n_sig, mu, sigma, n_bkg) = range(
    6
)  # the sum_dist array we will pass will be x_lo, x_hi, n_sig, mu, sigma, n_bkg
par_array = [(gaussian, [mu, sigma]), (uniform, [x_lo, x_hi])]

gauss_on_uniform = SumDists(
    par_array,
    [n_sig, n_bkg],
    "areas",
    parameter_names=["x_lo", "x_hi", "n_sig", "mu", "sigma", "n_bkg"],
    name="gauss_on_uniform",
)
