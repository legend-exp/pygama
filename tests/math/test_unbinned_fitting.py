import numpy as np
from pytest import approx

import pygama.math.unbinned_fitting as pgubf


def test_fit_unbinned():
    xr = (-2, 2)  # xrange

    rng = np.random.default_rng(42)
    mu = 1
    sigma = 0.1

    xdata = rng.normal(mu, sigma, size=1000)
    xdata = xdata[(xr[0] < xdata) & (xdata < xr[1])]

    n, xe = np.histogram(xdata, bins=50, range=xr)
    from pygama.math.functions.gauss import gaussian

    fit, fit_error, fit_cov = pgubf.fit_unbinned(
        gaussian.get_pdf, xdata, guess=[0, 0.9], cost_func="LL", extended=False
    )
    assert fit["mu"] == approx(mu, rel=1e-2)
    assert fit["sigma"] == approx(sigma, rel=1e-1)
