from __future__ import annotations

import lh5
import numpy as np
import pandas as pd

import pygama.pargen.lq_cal as lq
from pygama.math.distributions import gaussian


def test_lq_cal(lgnd_test_data):
    # test the HPGe calibration function
    # the function should return a calibration polynomial
    # that maps ADC channel to energy in keV

    # load test data here
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )

    data_df = lh5.read_as("ch1104000/dsp", data, "pd")

    data_df["cuspEmax_cal"] = data_df["cuspEmax"] * 0.155

    cal_dict = {
        "LQ_Ecorr": {
            "expression": "lq80/cuspEmax",
            "parameters": {},
        }
    }

    lqcal = lq.LQCal(
        cal_dict,
        "cuspEmax_cal",
        "dt_eff",
        lambda x: np.sqrt(1.5 + 0.1 * x),
        selection_string="index==index",
        cdf=gaussian,
        debug_mode=True,
    )

    data_df["LQ_Ecorr"] = np.divide(data_df["lq80"], data_df["cuspEmax"])

    lqcal.calibrate(data_df, "LQ_Ecorr")
    assert (lqcal.cut_val > 0) & (~np.isnan(lqcal.cut_val))
    assert (~np.isnan(lqcal.low_side_sf.loc[1592.50]["sf"])) & (
        lqcal.low_side_sf.loc[1592.50]["sf"] > 95
    )


def test_lq_cal_suffix(lgnd_test_data):
    # calibrating with a suffix must produce the suffixed output columns and
    # calibration expressions, leaving the unsuffixed names untouched
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )

    data_df = lh5.read_as("ch1104000/dsp", data, "pd")

    data_df["cuspEmax_cal"] = data_df["cuspEmax"] * 0.155

    cal_dict = {
        "LQ_Ecorr_alt": {
            "expression": "lq80/cuspEmax",
            "parameters": {},
        }
    }

    lqcal = lq.LQCal(
        cal_dict,
        "cuspEmax_cal",
        "dt_eff",
        lambda x: np.sqrt(1.5 + 0.1 * x),
        selection_string="index==index",
        cdf=gaussian,
        debug_mode=True,
    )

    data_df["LQ_Ecorr_alt"] = np.divide(data_df["lq80"], data_df["cuspEmax"])

    lqcal.calibrate(data_df, "LQ_Ecorr_alt", suffix="alt")
    assert (lqcal.cut_val > 0) & (~np.isnan(lqcal.cut_val))

    for name in ("LQ_Timecorr", "LQ_Corrected", "LQ_Classifier", "LQ_Cut"):
        assert f"{name}_alt" in cal_dict
        assert name not in cal_dict
    assert "LQ_Classifier_alt" in data_df
    assert "LQ_Cut_alt" in data_df
    assert cal_dict["LQ_Cut_alt"]["expression"] == "(LQ_Classifier_alt < a)"

    # the plot helpers must be steerable to the suffixed cut: only the
    # suffixed columns exist here, so with the default cut_param the queries
    # fail (swallowed) and the pass/fail histograms never get drawn
    fig = lq.plot_spectra(lqcal, data_df, cut_param="LQ_Cut_alt")
    assert len(fig.axes[0].patches) >= 3
    fig = lq.plot_sf_vs_energy(lqcal, data_df, cut_param="LQ_Cut_alt")
    assert fig.axes
    assert len(fig.axes[0].lines) == 1


def test_energy_width_correction_skips_pulser():
    # a pulser line with large LQ/E inside one energy window must be removed
    # by the selection, else that window's fit range estimate fails
    rng = np.random.default_rng(1)
    n, n_puls = 200_000, 5_000
    df = pd.DataFrame(
        {
            "cuspEmax_cal": np.r_[
                rng.uniform(250, 2650, n), rng.normal(916, 0.7, n_puls)
            ],
            "LQ_Dtcorr": np.r_[rng.normal(0.8, 0.2, n), rng.normal(5, 0.1, n_puls)],
            "is_pulser": np.r_[np.zeros(n, bool), np.ones(n_puls, bool)],
        }
    )

    lqcal = lq.LQCal(
        {},
        "cuspEmax_cal",
        "dt_eff",
        lambda x: np.sqrt(1.5 + 0.1 * x),
        selection_string="~is_pulser",
        cdf=gaussian,
        debug_mode=True,
    )
    lqcal.energy_width_correction(df, "LQ_Dtcorr", "cuspEmax_cal")

    assert len(lqcal.gauss_trends_hist) == 24
    assert np.all(np.isfinite(lqcal.mean_cor_res["popt"]))
    assert np.all(np.isfinite(lqcal.sig_cor_res["popt"]))
    assert df["LQ_Corrected"].notna().all()
