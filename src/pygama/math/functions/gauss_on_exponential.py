r"""
Provide a convenience function for a Gaussian on top of an exponential. The correct usage is :func:`gauss_on_exponential.pdf_ext(x, n_sig, mu, sigma, lambda, n_bkg, mu_exp, sigma_exp)`

Parameters
----------
n_sig
    The area of the Gaussian
mu, sigma
    The location and scale of the first Gaussian
n_bkg
    The area of the exponential
lambda
    The characteristic scale of the exponential
mu_exp, sigma_exp
    The location and scale of the exponential

Returns
-------
gauss_on_exponential
    A subclass of :class:`SumDists` and :class:`rv_continuous`, has methods of :func:`pdf`, :func:`cdf`, etc.
"""

from pygama.math.functions.exponential import exponential
from pygama.math.functions.gauss import gaussian
from pygama.math.functions.sum_dists import SumDists

(n_sig, mu, sigma, n_bkg, lamb, mu_exp, sigma_exp) = range(
    7
)  # the sum_dist array we will pass will be n_sig, mu, sigma, lambda, n_bkg, mu_exp, sigma_exp
par_array = [(gaussian, [mu, sigma]), (exponential, [mu_exp, sigma_exp, lamb])]

gauss_on_exponential = SumDists(
    par_array,
    [n_sig, n_bkg],
    "areas",
    parameter_names=["n_sig", "mu", "sigma", "n_bkg", "lambd", "mu_exp", "sigma_exp"],
    name="gauss_on_exponential",
)
