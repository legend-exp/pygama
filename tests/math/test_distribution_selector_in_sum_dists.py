import numpy as np
from pytest import approx

from pygama.math.functions.hpge_peak import hpge_peak

mu, sigma, tau, frac1, n_sig, hstep, lower_range, upper_range, n_bkg = range(9)

pars = [0, 1, 0.1, 0, 1, 0, np.inf, np.inf, 0]
cov = [
    [1e-16, 0, 0, 0, 0, 0, 0, 0, 0],  # dmu2
    [0, 1e-02, 0, 0, 0, 0, 0, 0, 0],  # dsig2
    [0, 0, 1e-16, 0, 0, 0, 0, 0, 0],  # dtau2
    [0, 0, 0, 1e-16, 0, 0, 0, 0, 0],  # dhtail2
    [0, 0, 0, 0, 1e-16, 0, 0, 0, 0],  # dnsig2
    [0, 0, 0, 0, 0, 1e-16, 0, 0, 0],  # dhstep2
    [0, 0, 0, 0, 0, 0, 1e-16, 0, 0],  # dlowerrange2
    [0, 0, 0, 0, 0, 0, 0, 1e-16, 0],  # dupperrange2
    [0, 0, 0, 0, 0, 0, 0, 0, 1e-16],  # dnbkg2
]


def test_get_mu():
    mu, mu_err = hpge_peak.get_mu(pars, cov)
    assert mu == 0
    assert mu_err == np.sqrt(1e-16)


def test_get_fwhm():
    fwhm, dfwhm = hpge_peak.get_fwhm(pars, cov)

    assert fwhm == approx(2.3548, rel=1e-5)
    assert dfwhm == approx(2.3548e-8, rel=1e-5)


def test_get_total_events():
    total_events, total_event_err = hpge_peak.get_total_events(pars, cov)

    assert total_events == 1
    assert total_event_err == approx(np.sqrt(2 * 1e-32))
