from __future__ import annotations

import logging

import lh5
import numpy as np
import pytest

from pygama.math.distributions import hpge_peak
from pygama.pargen import energy_cal


def test_peak_match():
    expected_peaks_kev = [1460, 2614.5]
    peaks_adu = [
        78.676315,
        288.4798,
        603.18506,
        1337.4973,
        2019.3586,
        3225.7288,
        3907.59,
        5795.8213,
        8470.816,
    ]
    _pars, best_ixtup, _best_iytup = energy_cal.poly_match(
        peaks_adu, expected_peaks_kev, deg=0, atol=10
    )
    assert np.array_equal(best_ixtup, [5, 7])


def test_hpge_cal(lgnd_test_data):
    # test the HPGe calibration function
    # the function should return a calibration polynomial
    # that maps ADC channel to energy in keV

    # load test data here
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )

    energy = lh5.read_as("ch1104000/dsp/cuspEmax", data, "np")

    glines = [860.564, 1592.53, 1620.50, 2103.53, 2614.50]

    pk_pars = [
        (860.564, (20, 20), hpge_peak),
        (1592.53, (20, 20), hpge_peak),
        (1620.50, (20, 20), hpge_peak),
        (2103.53, (20, 20), hpge_peak),
        (2614.50, (20, 20), hpge_peak),
    ]

    # test init
    cal = energy_cal.HPGeCalibration(
        "cuspEmax",
        glines,
        2615 / np.nanpercentile(energy, 99),
        deg=0,
        debug_mode=True,
    )

    # test dictionary generation
    out_dict = cal.gen_pars_dict()
    assert out_dict == {
        "expression": "a + b * cuspEmax",
        "parameters": {"a": 0.0, "b": 2615 / np.nanpercentile(energy, 99)},
    }

    cal.hpge_find_energy_peaks(energy, update_cal_pars=False)

    assert (cal.peaks_kev == glines).all()
    assert pytest.approx(cal.pars[1], 1) == 0.15
    assert cal.pars[0] == 0.0

    cal.hpge_find_energy_peaks(energy)

    assert len(cal.peaks_kev) == len(glines)
    assert (cal.peaks_kev == glines).all()
    assert pytest.approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0

    cal.hpge_get_energy_peaks(energy)

    assert len(cal.peaks_kev) == len(glines)
    assert (cal.peaks_kev == glines).all()
    assert pytest.approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0
    locs = cal.peak_locs.copy()
    cal.hpge_cal_energy_peak_tops(energy)

    assert len(cal.peaks_kev) == len(glines)
    assert (cal.peaks_kev == glines).all()
    assert pytest.approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0

    cal.peak_locs = locs
    cal.hpge_fit_energy_peaks(energy, peak_pars=pk_pars)

    assert len(cal.peaks_kev) == len(glines)
    assert (cal.peaks_kev == glines).all()
    assert pytest.approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0

    cal.get_energy_res_curve(
        energy_cal.FWHMLinear,
        interp_energy_kev={"Qbb": 2039.0},
    )

    assert (
        pytest.approx(
            cal.results["hpge_fit_energy_peaks"]["FWHMLinear"]["Qbb_fwhm_in_kev"], 0.1
        )
        == 2.3
    )


def test_hpge_cal_full_calibration(lgnd_test_data):
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )

    energy = lh5.read_as("ch1104000/dsp/cuspEmax", data, "np")

    glines = [860.564, 1592.53, 1620.50, 2103.53, 2614.50]

    pk_pars = [
        (860.564, (20, 20), hpge_peak),
        (1592.53, (20, 20), hpge_peak),
        (1620.50, (20, 20), hpge_peak),
        (2103.53, (20, 20), hpge_peak),
        (2614.50, (20, 20), hpge_peak),
    ]

    cal = energy_cal.HPGeCalibration(
        "cuspEmax",
        glines,
        2615 / np.nanpercentile(energy, 99),
        deg=0,
        debug_mode=True,
    )

    cal.full_calibration(energy, peak_pars=pk_pars)

    assert len(cal.peaks_kev) == len(glines)
    assert (cal.peaks_kev == glines).all()
    assert pytest.approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0


def test_hpge_cal_prominent_peak(lgnd_test_data):
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )

    energy = lh5.read_as("ch1104000/dsp/cuspEmax", data, "np")

    glines = [860.564, 1592.53, 1620.50, 2103.53, 2614.50]

    pk_pars = [
        (860.564, (20, 20), hpge_peak),
        (1592.53, (20, 20), hpge_peak),
        (1620.50, (20, 20), hpge_peak),
        (2103.53, (20, 20), hpge_peak),
        (2614.50, (20, 20), hpge_peak),
    ]

    # test in
    cal = energy_cal.HPGeCalibration(
        "cuspEmax",
        glines,
        2615 / np.nanpercentile(energy, 99),
        deg=0,
        debug_mode=True,
    )

    cal.calibrate_prominent_peak(energy, 2614.5, pk_pars)
    assert cal.peaks_kev[0] == 2614.5
    assert len(cal.peaks_kev) == 1
    assert pytest.approx(cal.pars[1], 0.1) == 0.15


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
def test_interpolate_energy_res_reports_unusable_draws(caplog):
    """The resolution model is bounded but the draw is not.

    ``sqrt(a + b*E)`` returns nan whenever a draw sends the radicand negative,
    and ``nanstd`` drops those silently; the count must be reported so a
    shrinking sample is visible rather than hidden.
    """
    # a sits close to its 0 bound with a wide error, so a sizeable fraction of
    # the draws go negative at a low interpolation energy
    results = {"parameters": [0.05, 1e-4], "cov": [[0.1**2, 0], [0, 1e-10]]}

    with caplog.at_level(logging.DEBUG, logger="pygama.pargen.energy_cal"):
        out = energy_cal.HPGeCalibration.interpolate_energy_res(
            energy_cal.FWHMLinear,
            np.array([100.0, 3000.0]),
            dict(results),
            interp_energy_kev={"low": 200.0},
        )

    assert "draws unusable" in caplog.text
    # the surviving sample still yields a finite uncertainty
    assert np.isfinite(out["low_fwhm_in_kev"])
    assert np.isfinite(out["low_fwhm_err_in_kev"])


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
def test_interpolate_energy_res_quiet_when_all_draws_usable(caplog):
    results = {"parameters": [4.0, 1e-3], "cov": [[0.01, 0], [0, 1e-10]]}

    with caplog.at_level(logging.DEBUG, logger="pygama.pargen.energy_cal"):
        energy_cal.HPGeCalibration.interpolate_energy_res(
            energy_cal.FWHMLinear,
            np.array([100.0, 3000.0]),
            dict(results),
            interp_energy_kev={"qbb": 2039.0},
        )

    assert "draws unusable" not in caplog.text
