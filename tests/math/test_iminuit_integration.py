import numpy as np
from iminuit import Minuit, cost
from pytest import approx
from scipy.stats import exponnorm

from pygama.math.distributions import exgauss, hpge_peak

xr = (-20, 20)  # xrange

tau = 1.1
sigma = 1
mu = 2
area = 1000
scipy_exgauss = exponnorm(tau / sigma, -mu, sigma)

xdata = scipy_exgauss.rvs(size=1000, random_state=1)
xmix = -1 * xdata

n, xe = np.histogram(xmix, bins=50, range=xr)
cx = 0.5 * (xe[1:] + xe[:-1])
dx = np.diff(xe)


def test_unbinned_nll():
    c = cost.UnbinnedNLL(xmix, exgauss.pdf_norm)

    m = Minuit(c, x_lo=-20, x_hi=20, tau=1, mu=0, sigma=3)
    m.fixed["x_lo", "x_hi"] = True
    m.limits["tau"] = (0.1, 3)
    m.limits["mu"] = (0, 2)
    m.limits["sigma"] = (0, 1.5)
    m.migrad()

    fit_tau = m.values["tau"]
    fit_mu = m.values["mu"]
    fit_sigma = m.values["sigma"]

    assert tau == approx(fit_tau, 0.1)
    assert mu == approx(fit_mu, 0.1)
    assert sigma == approx(fit_sigma, 0.1)


def test_extended_unbinned_nll():
    c = cost.ExtendedUnbinnedNLL(xmix, exgauss.pdf_ext)

    m = Minuit(c, x_lo=-7, x_hi=7, area=100, mu=0, sigma=3, tau=1)
    m.fixed["x_lo", "x_hi"] = True
    m.limits["tau", "mu", "sigma"] = (0.01, 5)
    m.limits["area"] = (0, 2000)
    m.migrad()

    fit_area = m.values["area"]
    fit_tau = m.values["tau"]
    fit_mu = m.values["mu"]
    fit_sigma = m.values["sigma"]

    assert area == approx(fit_area, 0.1)
    assert tau == approx(fit_tau, 0.1)
    assert mu == approx(fit_mu, 0.1)
    assert sigma == approx(fit_sigma, 0.1)


def test_binned_nll():
    c = cost.BinnedNLL(n, xe, exgauss.cdf_norm)

    m = Minuit(c, x_lo=-20, x_hi=20, mu=0, sigma=3, tau=1)
    m.fixed["x_lo", "x_hi"] = True
    m.limits["tau"] = (0.1, 3)
    m.limits["mu"] = (0, 2)
    m.limits["sigma"] = (0, 1.5)
    m.migrad()

    fit_tau = m.values["tau"]
    fit_mu = m.values["mu"]
    fit_sigma = m.values["sigma"]

    assert tau == approx(fit_tau, 0.1)
    assert mu == approx(fit_mu, 0.1)
    assert sigma == approx(fit_sigma, 0.1)


def test_extended_binned_nll():
    c = cost.ExtendedBinnedNLL(n, xe, exgauss.cdf_ext)

    m = Minuit(c, area=100, mu=0, sigma=3, tau=1)
    m.limits["tau", "mu", "sigma"] = (0.01, 5)
    m.limits["area"] = (0, 2000)

    m.migrad()

    fit_area = m.values["area"]
    fit_tau = m.values["tau"]
    fit_mu = m.values["mu"]
    fit_sigma = m.values["sigma"]

    assert area == approx(fit_area, 0.1)
    assert tau == approx(fit_tau, 0.1)
    assert mu == approx(fit_mu, 0.1)
    assert sigma == approx(fit_sigma, 0.1)


def test_name_introspection():
    c = cost.ExtendedUnbinnedNLL(xmix, hpge_peak.pdf_ext)
    m = Minuit(c, xr[0], xr[-1], 100, 1, 2, 0.1, 20, 10, 0.1)

    m.values["n_sig"] = 0
    m.values["mu"] = 0
    m.values["sigma"] = 0
    m.values["htail"] = 0
    m.values["tau"] = 0
    m.values["n_bkg"] = 0
    m.values["hstep"] = 0
    m.fixed["x_lo"] = 0
    m.fixed["x_hi"] = 0
