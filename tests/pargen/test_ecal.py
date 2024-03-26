import lgdo.lh5 as lh5
import numpy as np
from pytest import approx

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
    pars, best_ixtup, best_iytup = energy_cal.poly_match(
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
    assert approx(cal.pars[1], 1) == 0.15
    assert cal.pars[0] == 0.0

    cal.hpge_find_energy_peaks(energy)

    assert len(cal.peaks_kev) == len(glines) and (cal.peaks_kev == glines).all()
    assert approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0

    cal.hpge_get_energy_peaks(energy)

    assert len(cal.peaks_kev) == len(glines) and (cal.peaks_kev == glines).all()
    assert approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0
    locs = cal.peak_locs.copy()
    cal.hpge_cal_energy_peak_tops(energy)

    assert len(cal.peaks_kev) == len(glines) and (cal.peaks_kev == glines).all()
    assert approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0

    cal.peak_locs = locs
    cal.hpge_fit_energy_peaks(energy, peak_pars=pk_pars)

    assert len(cal.peaks_kev) == len(glines) and (cal.peaks_kev == glines).all()
    assert approx(cal.pars[1], 0.1) == 0.15
    assert cal.pars[0] == 0.0

    cal.get_energy_res_curve(
        energy_cal.FWHMLinear,
        interp_energy_kev={"Qbb": 2039.0},
    )

    assert (
        approx(
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

    assert len(cal.peaks_kev) == len(glines) and (cal.peaks_kev == glines).all()
    assert approx(cal.pars[1], 0.1) == 0.15
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
    assert cal.peaks_kev[0] == 2614.5 and len(cal.peaks_kev) == 1
    assert approx(cal.pars[1], 0.1) == 0.15
