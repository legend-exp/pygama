import numpy as np

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
