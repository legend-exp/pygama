r"""
Provide a convenience function for three gaussians on two steps

Parameters
----------
x_lo
    The lower range to compute the normalization of the step functions
x_hi
    The upper range to compute the normalization of the step functions
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

Example
-------
triple_gauss_on_double_step.get_pdf(x, pars = [x_lo, x_hi, n_sig1, mu1, sigma1, n_sig2, mu2, sigma2, n_sig3, mu3, sigma3, n_bkg1, hstep1, n_bkg2, hstep2])

Returns
-------
triple_gauss_on_double_step
    A subclass of SumDists and rv_continuous, has methods of pdf, cdf, etc.

Notes
-----
The first step function shares the mu_1, sigma_1 with the first Gaussian,
and the second step function shares the mu_2, sigma_2 with the second Gaussian
"""

from pygama.math.functions.gauss import gaussian
from pygama.math.functions.gauss_on_step import gauss_on_step
from pygama.math.functions.sum_dists import SumDists

(x_lo, x_hi, area1, mu1, sigma1, area4, hstep1, area2, mu2, sigma2, area5, hstep2) = (
    range(12)
)
par_array = [
    (gauss_on_step, [x_lo, x_hi, area1, mu1, sigma1, area4, hstep1]),
    (gauss_on_step, [x_lo, x_hi, area2, mu2, sigma2, area5, hstep2]),
]

double_gauss_on_double_step = SumDists(
    par_array,
    [],
    None,
    parameter_names=[
        "x_lo",
        "x_hi",
        "area1",
        "mu1",
        "sigma1",
        "area4",
        "hstep1",
        "area2",
        "mu2",
        "sigma2",
        "area5",
        "hstep2",
    ],
)

(
    x_lo,
    x_hi,
    area1,
    mu1,
    sigma1,
    area2,
    mu2,
    sigma2,
    area3,
    mu3,
    sigma3,
    area4,
    hstep1,
    area5,
    hstep2,
) = range(15)

par_array = [
    (gaussian, [mu3, sigma3]),
    (
        double_gauss_on_double_step,
        [
            x_lo,
            x_hi,
            area1,
            mu1,
            sigma1,
            area4,
            hstep1,
            area2,
            mu2,
            sigma2,
            area5,
            hstep2,
        ],
    ),
]
triple_gauss_on_double_step = SumDists(
    par_array,
    [area3],
    "one_area",
    parameter_names=[
        "x_lo",
        "x_hi",
        "n_sig1",
        "mu1",
        "sigma1",
        "n_sig2",
        "mu2",
        "sigma2",
        "n_sig3",
        "mu3",
        "sigma3",
        "n_bkg1",
        "hstep1",
        "n_bkg2",
        "hstep2",
    ],
    name="triple_gauss_on_double_step",
)
