from __future__ import annotations

import logging

import numpy as np
import pytest

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
    _amp, _mu, sig, htail, tau, _bg0, _hstep = pars
    fwhm, dfwhm = pgb.hpge_peak_fwhm(sig, htail, tau, cov)
    assert fwhm == pytest.approx(2.3548, rel=1e-5)
    assert dfwhm == pytest.approx(2.3548e-1, rel=1e-5)


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

    _amp, _mu, sig, htail, tau, _bg0, _hstep = pars
    fwhm, dfwhm = pgb.hpge_peak_fwhm(sig, htail, tau, cov)
    assert fwhm == pytest.approx(np.log(2), rel=1e-5)
    assert dfwhm == pytest.approx(np.log(2) / 10, rel=1e-5)


def test_bootstrap_valid_pars_redraws_at_boundary():
    # mean sits on the htail == 0 bound, so ~half of an unconstrained draw is
    # unusable; the sampler must redraw rather than return a short sample
    pars = [1, 0, 1, 0, 0.1, 0, 0]
    cov = np.diag([1e-16, 1e-16, 1e-16, 1e-2, 1e-16, 1e-16, 1e-16])

    def evaluate(p):
        if p[3] < 0:
            msg = "htail outside allowed limits of 0 and 1"
            raise ValueError(msg)
        return p[3]

    rng = np.random.default_rng(1)
    accepted, values, n_drawn, last_error = pgb.bootstrap_valid_pars(
        rng, pars, cov, evaluate, size=100
    )

    assert len(values) == 100
    assert len(accepted) == 100
    assert np.all(accepted[:, 3] >= 0)
    assert np.all(np.isfinite(values))
    # roughly half the draws are rejected, so more than `size` were needed
    assert n_drawn > 100
    assert "htail outside allowed limits" in str(last_error)


def test_bootstrap_valid_pars_rejects_non_finite():
    # hpge_peak_mode signals an out-of-range htail by returning nan instead of
    # raising; such draws must be rejected too, not averaged in
    pars = [1, 0, 1, 0, 0.1, 0, 0]
    cov = np.diag([1e-16, 1e-16, 1e-16, 1e-2, 1e-16, 1e-16, 1e-16])

    rng = np.random.default_rng(1)
    accepted, values, n_drawn, last_error = pgb.bootstrap_valid_pars(
        rng, pars, cov, lambda p: p[3] if p[3] >= 0 else np.nan, size=50
    )

    assert len(values) == 50
    assert np.all(np.isfinite(values))
    assert np.all(accepted[:, 3] >= 0)
    assert n_drawn > 50
    assert last_error == "non-finite value"


def test_bootstrap_valid_pars_respects_bounds():
    # htail is bounded (0, 0.5) by the fit, but hpge_peak_fwfm only rejects
    # htail > 1 -- draws in between must be excluded by the bounds map
    pars = [1, 0, 1, 0.45, 0.1, 0, 0]
    cov = np.diag([1e-16, 1e-16, 1e-16, 1e-2, 1e-16, 1e-16, 1e-16])
    bounds = {3: (0, 0.5)}

    rng = np.random.default_rng(1)
    accepted, values, n_drawn, last_error = pgb.bootstrap_valid_pars(
        rng, pars, cov, lambda p: p[3], size=100, bounds=bounds
    )

    assert len(values) == 100
    assert np.all(accepted[:, 3] >= 0)
    assert np.all(accepted[:, 3] <= 0.5)
    # without the bounds the same draws would have been accepted up to 1.0
    assert n_drawn > 100
    assert last_error == "outside fit bounds"

    rng = np.random.default_rng(1)
    unbounded, _, _, _ = pgb.bootstrap_valid_pars(
        rng, pars, cov, lambda p: p[3], size=100
    )
    assert np.any(unbounded[:, 3] > 0.5)


def test_bootstrap_valid_pars_open_sided_bounds():
    pars = [1, 0, 1, 0.1, 0.5, 0, 0]
    cov = np.diag([1e-16, 1e-16, 1e-16, 1e-16, 1e-2, 1e-16, 1e-16])

    rng = np.random.default_rng(1)
    accepted, values, _, _ = pgb.bootstrap_valid_pars(
        rng, pars, cov, lambda p: p[4], size=50, bounds={4: (0, None)}
    )

    assert len(values) == 50
    assert np.all(accepted[:, 4] >= 0)


def test_bootstrap_valid_pars_terminates_when_nothing_valid():
    pars = [1, 0, 1, 0, 0.1, 0, 0]
    cov = np.diag([1e-16] * 7)

    def always_raises(p):
        msg = "no draw is usable"
        raise ValueError(msg)

    rng = np.random.default_rng(1)
    accepted, values, n_drawn, last_error = pgb.bootstrap_valid_pars(
        rng, pars, cov, always_raises, size=10, max_draws=30
    )

    assert len(values) == 0
    assert accepted.shape == (0, len(pars))
    assert n_drawn == 30
    assert "no draw is usable" in str(last_error)


def test_hpge_peak_mode_drops_and_reports_unusable_draws(caplog):
    """The mode bootstrap drops invalid draws (no redraw) but must report them."""
    cov = np.diag([1e-16, 1e-4, 1e-4, 0.02**2, 1e-2, 1e-16, 1e-16])

    with caplog.at_level(logging.DEBUG, logger="pygama.math.hpge_peak_fitting"):
        mode, mode_err = pgb.hpge_peak_mode(0.0, 1.0, 0.0, 5.0, cov=cov)

    assert np.isfinite(mode)
    assert np.isfinite(mode_err)
    assert mode_err > 0
    # htail sits on the bound, so about half the draws are unusable and the
    # count must appear in the log rather than vanishing into nanstd
    assert "bootstrap draws unusable" in caplog.text

    # a fit well away from the bound wastes no draws and logs nothing
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="pygama.math.hpge_peak_fitting"):
        pgb.hpge_peak_mode(0.0, 1.0, 0.2, 5.0, cov=cov)
    assert "bootstrap draws unusable" not in caplog.text


def test_hpge_peak_fwfm_boundary_error_is_unbiased():
    """A tail fraction near zero must not shrink the bootstrap sample."""
    sigma, tau = 1.0, 5.0
    cov = np.diag([1e-16, 1e-16, 1e-4, 0.02**2, 1e-2, 1e-16, 1e-16])

    # htail well away from the bound: no draw is rejected, so the result must
    # be identical to the plain unconstrained bootstrap
    _, err_far = pgb.hpge_peak_fwfm(sigma, 0.2, tau, frac_max=0.5, cov=cov)
    assert np.isfinite(err_far)
    assert err_far > 0

    # htail within 1 sigma of zero: draws are rejected and redrawn, and the
    # resulting spread is larger than the one-sidedly truncated estimate
    _, err_near = pgb.hpge_peak_fwfm(sigma, 0.02, tau, frac_max=0.5, cov=cov)
    assert np.isfinite(err_near)

    rng = np.random.default_rng(1)
    truncated = np.array(
        [
            pgb.hpge_peak_fwfm(p[2], p[3], p[4], frac_max=0.5)
            if 0 <= p[3] <= 1
            else np.nan
            for p in rng.multivariate_normal([1, 0, sigma, 0.02, tau, 0, 0], cov, 100)
        ]
    )
    assert np.count_nonzero(np.isnan(truncated)) > 0
    assert err_near > np.nanstd(truncated)
