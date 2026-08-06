"""Tests for the ``use_log_pdf`` mode of the A/E and survival-fraction fits."""

from __future__ import annotations

import numpy as np

from pygama.pargen.AoE_cal import unbinned_aoe_fit
from pygama.pargen.survival_fractions import get_survival_fraction


def test_unbinned_aoe_fit_log_mode_agrees():
    rng = np.random.default_rng(2718)
    n = 20000
    aoe = np.concatenate(
        [
            rng.normal(0.0, 0.01, int(n * 0.9)),
            -rng.exponential(0.03, int(n * 0.1)),
        ]
    )

    res = {}
    for use_log in (False, True):
        pars, errs, _cov, (_, _, _, _, _, _, valid, _m) = unbinned_aoe_fit(
            aoe, use_log_pdf=use_log
        )
        assert valid
        res[use_log] = (np.array(pars), np.array(errs))

    pars_std, errs_std = res[False]
    pars_log, errs_log = res[True]
    # mu / sigma are the physically used outputs
    names = ["x_lo", "x_hi", "n_sig", "mu", "sigma", "n_bkg", "tau"]
    mu_i, sigma_i = names.index("mu"), names.index("sigma")
    # mu sits near zero for normalised A/E, so measure its shift in units of
    # the peak width rather than relative to itself
    assert abs(pars_log[mu_i] - pars_std[mu_i]) < 1e-4 * pars_std[sigma_i]
    assert np.isclose(pars_log[sigma_i], pars_std[sigma_i], rtol=1e-4)
    assert np.allclose(pars_log, pars_std, rtol=5e-3, atol=1e-12)
    assert np.allclose(errs_log, errs_std, rtol=5e-2, atol=1e-12)


def test_get_survival_fraction_log_mode_agrees():
    rng = np.random.default_rng(31415)
    n_sig, n_bkg = 5000, 1000
    mu, sigma = 1592.5, 1.2
    energy = np.concatenate(
        [
            rng.normal(mu, sigma, n_sig),
            rng.uniform(mu - 30, mu + 30, n_bkg),
        ]
    )
    # cut parameter correlated with nothing: signal survives ~90%, bkg ~50%
    cut_param = np.concatenate(
        [
            rng.normal(2.0, 1.0, n_sig),
            rng.normal(0.0, 1.0, n_bkg),
        ]
    )

    res = {}
    for use_log in (False, True):
        sf, sf_err, _, _ = get_survival_fraction(
            energy,
            cut_param,
            0.0,
            mu,
            2.355 * sigma,
            fit_range=(mu - 30, mu + 30),
            mode="greater",
            use_log_pdf=use_log,
        )
        assert np.isfinite(sf)
        res[use_log] = (sf, sf_err)

    assert np.isclose(res[True][0], res[False][0], rtol=1e-3)
    assert np.isclose(res[True][1], res[False][1], rtol=5e-2)
