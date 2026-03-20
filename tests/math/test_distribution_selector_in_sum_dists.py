from __future__ import annotations

import numpy as np
import pytest

from pygama.math.functions.hpge_peak import hpge_peak

lower_range, upper_range, n_sig, mu, sigma, frac1, tau, n_bkg, hstep = range(9)
pars = [np.inf, np.inf, 1, 0, 1, 0, 0.1, 0, 0]
cov = [
    [1e-16, 0, 0, 0, 0, 0, 0, 0, 0],  # dxlo2
    [0, 1e-16, 0, 0, 0, 0, 0, 0, 0],  # dxhi2
    [0, 0, 1e-16, 0, 0, 0, 0, 0, 0],  # damp2
    [0, 0, 0, 1e-16, 0, 0, 0, 0, 0],  # dmu2
    [0, 0, 0, 0, 1e-02, 0, 0, 0, 0],  # dsig2
    [0, 0, 0, 0, 0, 1e-16, 0, 0, 0],  # dhtail2
    [0, 0, 0, 0, 0, 0, 1e-16, 0, 0],  # dtau2
    [0, 0, 0, 0, 0, 0, 0, 1e-16, 0],  # dbg02
    [0, 0, 0, 0, 0, 0, 0, 0, 1e-16],  # dhs2
]


def test_get_mu():
    mu, mu_err = hpge_peak.get_mu(pars, cov)
    assert mu == 0
    assert mu_err == np.sqrt(1e-16)


def test_get_fwhm():
    fwhm, dfwhm = hpge_peak.get_fwhm(pars, cov)

    assert fwhm == pytest.approx(2.3548, rel=1e-5)
    assert dfwhm == pytest.approx(2.3548e-1, rel=1e-5)


def test_get_total_events():
    total_events, total_event_err = hpge_peak.get_total_events(pars, cov)

    assert total_events == 1
    assert total_event_err == pytest.approx(np.sqrt(2 * 1e-32))
