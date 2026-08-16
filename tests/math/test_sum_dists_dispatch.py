"""Regression tests for the cached signature-wrapper dispatch of SumDists.

The wrappers returned by attribute access are cached per instance; these tests
pin down that the cache changes neither the values produced, the advertised
signatures, nor runtime-state handling (``components``), and that the norm
computed with an explicit subtraction matches the previous ``np.diff`` form.
"""

from __future__ import annotations

import copy
import inspect

import numpy as np

from pygama.math.distributions import gauss_on_step, hpge_peak
from pygama.math.functions.sum_dists import get_areas_fracs

X = np.linspace(13000.0, 13140.0, 501)
# x_lo, x_hi, n_sig, mu, sigma, htail, tau, n_bkg, hstep
HPGE_PARS = (13000.0, 13140.0, 900.0, 13072.0, 17.5, 0.1, 40.0, 100.0, 0.01)
# x_lo, x_hi, n_sig, mu, sigma, n_bkg, hstep
STEP_PARS = (13000.0, 13140.0, 900.0, 13072.0, 17.5, 100.0, 0.01)


def test_wrapper_forwards_exactly():
    for dist, pars in ((hpge_peak, HPGE_PARS), (gauss_on_step, STEP_PARS)):
        for method in ("get_pdf", "get_cdf", "pdf_norm", "cdf_norm"):
            wrapped = getattr(dist, method)
            plain = object.__getattribute__(dist, method)
            assert np.array_equal(wrapped(X, *pars), plain(X, *pars))
        sig_w, probs_w = dist.pdf_ext(X, *pars)
        sig_p, probs_p = object.__getattribute__(dist, "pdf_ext")(X, *pars)
        assert sig_w == sig_p
        assert np.array_equal(probs_w, probs_p)


def test_wrapper_is_cached_and_signature_preserved():
    assert hpge_peak.get_pdf is hpge_peak.get_pdf
    assert hpge_peak.pdf_ext is hpge_peak.pdf_ext
    assert hpge_peak.get_pdf is not hpge_peak.get_cdf

    assert list(inspect.signature(hpge_peak.get_pdf).parameters) == list(
        hpge_peak.x_shapes
    )
    assert list(inspect.signature(hpge_peak.pdf_ext).parameters) == list(
        hpge_peak.extended_shapes
    )


def test_norm_matches_np_diff_reference():
    params = np.array(HPGE_PARS)
    fracs, areas = get_areas_fracs(
        params,
        hpge_peak.area_frac_idxs,
        hpge_peak.frac_flag,
        hpge_peak.area_flag,
        hpge_peak.one_area_flag,
    )
    support = np.array([HPGE_PARS[0], HPGE_PARS[1]])
    ref_sig = (
        areas[0]
        * fracs[0]
        * np.diff(hpge_peak.dists[0].get_cdf(support, *params[hpge_peak.par_idxs[0]]))[
            0
        ]
        + areas[1]
        * fracs[1]
        * np.diff(hpge_peak.dists[1].get_cdf(support, *params[hpge_peak.par_idxs[1]]))[
            0
        ]
    )
    sig, _ = hpge_peak.pdf_ext(X, *HPGE_PARS)
    assert sig == ref_sig

    ref_norm = np.diff(hpge_peak.get_cdf(support, *HPGE_PARS))
    ref_pdf_norm = hpge_peak.get_pdf(X, *HPGE_PARS) / ref_norm
    assert np.array_equal(hpge_peak.pdf_norm(X, *HPGE_PARS), ref_pdf_norm)
    ref_cdf_norm = hpge_peak.get_cdf(X, *HPGE_PARS) / ref_norm
    assert np.array_equal(hpge_peak.cdf_norm(X, *HPGE_PARS), ref_cdf_norm)


def test_components_toggle_still_honoured():
    sig, probs = hpge_peak.pdf_ext(X, *HPGE_PARS)
    hpge_peak.components = True
    try:
        sig_c, comp_1, comp_2 = hpge_peak.pdf_ext(X, *HPGE_PARS)
    finally:
        hpge_peak.components = False
    assert sig_c == sig
    assert np.array_equal(comp_1 + comp_2, probs)


def test_deepcopy_gets_independent_cache():
    dist = copy.deepcopy(gauss_on_step)
    assert dist.get_pdf is not gauss_on_step.get_pdf
    assert np.array_equal(
        dist.get_pdf(X, *STEP_PARS), gauss_on_step.get_pdf(X, *STEP_PARS)
    )
