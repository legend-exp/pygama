import numpy as np
from pytest import approx

import pygama.math.peak_fitting as pgf


def test_mostly_gauss_fwhm():
    pars = [0, 1, 0, 0, 0.1, 0, 1]
    cov = [
        [1e-16, 0, 0, 0, 0, 0, 0],  # dmu2
        [0, 1e-02, 0, 0, 0, 0, 0],  # dsig2
        [0, 0, 1e-16, 0, 0, 0, 0],  # dhstep2
        [0, 0, 0, 1e-16, 0, 0, 0],  # dhtail2
        [0, 0, 0, 0, 1e-16, 0, 0],  # dtau2
        [0, 0, 0, 0, 0, 1e-16, 0],  # dbg02
        [0, 0, 0, 0, 0, 0, 1e-16],
    ]  # damp2
    mu, sig, hstep, htail, tau, bg0, amp = pars
    fwhm, dfwhm = pgf.radford_fwhm(sig, htail, tau, cov)
    assert fwhm == approx(2.3548, rel=1e-5)
    assert dfwhm == approx(2.3548e-8, rel=1e-5)


def test_mostly_exp_fwhm():
    pars = [0, 1e-6, 0, 1, 1, 0, 1]
    cov = [
        [1e-16, 0, 0, 0, 0, 0, 0],  # dmu2
        [0, 1e-16, 0, 0, 0, 0, 0],  # dsig2
        [0, 0, 1e-16, 0, 0, 0, 0],  # dhstep2
        [0, 0, 0, 1e-16, 0, 0, 0],  # dhtail2
        [0, 0, 0, 0, 1e-02, 0, 0],  # dtau2
        [0, 0, 0, 0, 0, 1e-16, 0],  # dbg02
        [0, 0, 0, 0, 0, 0, 1e-16],
    ]  # damp2
    mu, sig, hstep, htail, tau, bg0, amp = pars
    fwhm, dfwhm = pgf.radford_fwhm(sig, htail, tau, cov)
    assert fwhm == approx(np.log(2), rel=1e-5)
    assert dfwhm == approx(np.log(2) / 10, rel=1e-5)
