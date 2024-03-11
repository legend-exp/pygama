import numpy as np
from pytest import approx

import pygama.math.hpge_peak_fitting as pgb


def test_mostly_gauss_fwhm():
    # parameters need to be in order of n_sig, mu, sigma, htail, tau, n_bkg, hstep
    pars = [1, 0, 1, 0, 0.1, 0, 0]
    cov = [
        [1e-16, 0, 0, 0, 0, 0, 0],  # damp2
        [0, 1e-16, 0, 0, 0, 0, 0],  # dmu2
        [0, 0, 1e-02, 0, 0, 0, 0],  # dsig2
        [0, 0, 0, 1e-16, 0, 0, 0],  # dhtail2
        [0, 0, 0, 0, 1e-16, 0, 0],  # dtau2
        [0, 0, 0, 0, 0, 1e-16, 0],  # dbg02
        [0, 0, 0, 0, 0, 0, 1e-16],  # dhstep2
    ]
    amp, mu, sig, htail, tau, bg0, hstep = pars
    fwhm, dfwhm = pgb.hpge_peak_fwhm(sig, htail, tau, cov)
    assert fwhm == approx(2.3548, rel=1e-5)
    assert dfwhm == approx(2.3548e-1, rel=1e-5)


def test_mostly_exp_fwhm():
    # parameters need to be in order of n_sig, mu, sigma, htail, tau, n_bkg, hstep
    pars = [1, 0, 1e-6, 1, 1, 0, 0]
    cov = [
        [1e-16, 0, 0, 0, 0, 0, 0],  # damp2
        [0, 1e-16, 0, 0, 0, 0, 0],  # dmu2
        [0, 0, 1e-16, 0, 0, 0, 0],  # dsig2
        [0, 0, 0, 1e-16, 0, 0, 0],  # dhtail2
        [0, 0, 0, 0, 1e-02, 0, 0],  # dtau2
        [0, 0, 0, 0, 0, 1e-16, 0],  # dbg02
        [0, 0, 0, 0, 0, 0, 1e-16],  # dhs2
    ]

    amp, mu, sig, htail, tau, bg0, hstep = pars
    fwhm, dfwhm = pgb.hpge_peak_fwhm(sig, htail, tau, cov)
    assert fwhm == approx(np.log(2), rel=1e-5)
    assert dfwhm == approx(np.log(2) / 10, rel=1e-5)
