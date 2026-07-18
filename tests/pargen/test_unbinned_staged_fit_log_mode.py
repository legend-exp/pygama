"""Tests for the ``use_log_pdf`` mode of ``unbinned_staged_energy_fit``.

The log mode hands ``iminuit`` the log-density directly (``log=True``), which
replaces the sorted log-sum with a plain sum.  The fit result must agree with
the standard mode to within float summation noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from lgdo import Array, Table

import pygama.math.distributions as pgf
from pygama.math.functions.sum_dists import _TINY_FLOAT
from pygama.math.hpge_peak_fitting import hpge_peak_fwfm
from pygama.pargen.energy_cal import unbinned_staged_energy_fit
from pygama.pargen.energy_optimisation import fom_fwhm_no_alpha_sweep

RNG = np.random.default_rng(8175)
MU, SIGMA = 13072.0, 17.5
N_SIG, N_TAIL, N_BKG = 3000, 300, 400
ENERGY = np.concatenate(
    [
        RNG.normal(MU, SIGMA, N_SIG),
        # low-side exponential tail so htail/tau are well constrained
        MU - RNG.exponential(40.0, N_TAIL) - RNG.normal(0, SIGMA, N_TAIL),
        RNG.uniform(MU - 250, MU + 250, N_BKG),
    ]
)
FIT_RANGE = (MU - 250, MU + 250)


def test_log_pdf_ext_matches_pdf_ext():
    x = np.linspace(*FIT_RANGE, 1001)
    pars = (FIT_RANGE[0], FIT_RANGE[1], 900.0, MU, SIGMA, 0.1, 40.0, 100.0, 0.01)
    sig, probs = pgf.hpge_peak.pdf_ext(x, *pars)
    sig_log, log_probs = pgf.hpge_peak.log_pdf_ext(x, *pars)
    assert sig_log == sig
    assert np.array_equal(log_probs, np.log(probs + _TINY_FLOAT))


def test_staged_fit_log_mode_agrees_with_standard():
    res = {}
    for use_log in (False, True):
        pars, errs, _cov, _csqr, _func, _mask, valid, _m = unbinned_staged_energy_fit(
            ENERGY,
            func=pgf.hpge_peak,
            fit_range=FIT_RANGE,
            use_log_pdf=use_log,
        )
        assert valid
        res[use_log] = (np.array(pars), np.array(errs))

    pars_std, errs_std = res[False]
    pars_log, errs_log = res[True]
    assert np.allclose(pars_log, pars_std, rtol=2e-3, atol=1e-10)
    assert np.allclose(errs_log, errs_std, rtol=2e-2, atol=1e-10)

    # the physical observable must agree much tighter than the (partially
    # degenerate) shape parameters: pars = (x_lo, x_hi, n_sig, mu, sigma,
    # htail, tau, n_bkg, hstep)
    fwhm_std = hpge_peak_fwfm(pars_std[4], pars_std[5], pars_std[6])
    fwhm_log = hpge_peak_fwfm(pars_log[4], pars_log[5], pars_log[6])
    assert np.isclose(fwhm_log, fwhm_std, rtol=1e-4)


def test_use_log_pdf_requires_log_pdf_ext():
    class NoLogPdf:
        pass

    with pytest.raises(ValueError, match="log_pdf_ext"):
        unbinned_staged_energy_fit(ENERGY, func=NoLogPdf(), use_log_pdf=True)


def test_hpge_calibration_log_mode_agrees(lgnd_test_data):
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )
    import lh5

    from pygama.pargen import energy_cal

    energy = lh5.read_as("ch1104000/dsp/cuspEmax", data, "np")
    glines = [860.564, 1592.53, 1620.50, 2103.53, 2614.50]
    pk_pars = [(line, (20, 20), pgf.hpge_peak) for line in glines]

    res = {}
    for use_log in (False, True):
        cal = energy_cal.HPGeCalibration(
            "cuspEmax",
            glines,
            2615 / np.nanpercentile(energy, 99),
            deg=0,
            debug_mode=True,
        )
        cal.hpge_find_energy_peaks(energy)
        cal.hpge_get_energy_peaks(energy)
        cal.hpge_fit_energy_peaks(energy, peak_pars=pk_pars, use_log_pdf=use_log)
        cal.get_energy_res_curve(
            energy_cal.FWHMLinear, interp_energy_kev={"Qbb": 2039.0}
        )
        fit_results = cal.results["hpge_fit_energy_peaks"]
        res[use_log] = (
            np.array(cal.pars),
            np.array(fit_results["pk_pos"], dtype=float),
            fit_results["FWHMLinear"]["Qbb_fwhm_in_kev"],
        )

    pars_std, pos_std, qbb_std = res[False]
    pars_log, pos_log, qbb_log = res[True]
    assert np.allclose(pars_log, pars_std, rtol=1e-6)
    assert np.allclose(pos_log, pos_std, rtol=1e-6)
    assert np.isclose(qbb_log, qbb_std, rtol=1e-3)


def test_fom_wiring_forwards_use_log_pdf():
    tb = Table(col_dict={"cuspEmax": Array(ENERGY)})
    kd = {
        "parameter": "cuspEmax",
        "func": pgf.hpge_peak,
        "peak": 2614.5,
        "kev_width": (50, 50),
    }
    res = {
        use_log: fom_fwhm_no_alpha_sweep(tb, kd, alpha=0, use_log_pdf=use_log)
        for use_log in (False, True)
    }
    assert np.isfinite(res[True]["fwhm"])
    assert np.isclose(res[True]["fwhm"], res[False]["fwhm"], rtol=1e-4)
