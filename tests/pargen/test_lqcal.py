from __future__ import annotations

import lh5
import numpy as np
import pytest

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

    # data behind the plots, as saved by the dataflow instead of the figures
    spec = lq.get_spectra_data(lqcal, data_df, cut_param="LQ_Cut_alt")
    sel = data_df.query(lqcal.selection_string)
    expected, _ = np.histogram(
        sel.query("LQ_Cut_alt")["cuspEmax_cal"], bins=spec["edges"]
    )
    np.testing.assert_array_equal(spec["after_cut"], expected)
    assert (spec["before"] == spec["after_cut"] + spec["rejected"]).all()

    sf = lq.get_sf_vs_energy_data(lqcal, data_df, cut_param="LQ_Cut_alt")
    assert ((sf["sf"] >= 0) & (sf["sf"] <= 100)).all()
    hist = lq.get_classifier_data(lqcal, data_df, lq_param="LQ_Classifier_alt")
    assert hist["counts"].shape == (699, 499)
    dt = lq.get_drift_time_correction_data(lqcal, data_df, lq_param="LQ_Timecorr_alt")
    assert dt["counts"].shape == (100, 100)
    assert len(dt["dt_range"]) == 2
    cut = lq.get_lq_cut_fit_data(lqcal, data_df)
    assert len(cut["counts"]) == len(cut["edges"]) - 1
    curves = lq.get_survival_fraction_curves_data(lqcal, data_df)
    assert curves["cut_val"] == lqcal.cut_val
    assert curves["peaks"]
    for plot in (
        lq.plot_spectra,
        lq.plot_sf_vs_energy,
        lq.plot_classifier,
        lq.plot_drift_time_correction,
        lq.plot_lq_cut_fit,
        lq.plot_survival_fraction_curves,
    ):
        assert plot.data_func is not None


@pytest.mark.filterwarnings("ignore:No artists with labels:UserWarning")
def test_lq_plots_without_calibration():
    # plotting must never raise when calibration steps were skipped or failed:
    # a failing plot aborts the whole dataflow job
    import matplotlib as mpl
    import pandas as pd

    mpl.use("Agg")
    lqcal = lq.LQCal(
        {}, "cuspEmax_cal", "dt_eff", lambda x: np.sqrt(1.5 + 0.1 * x),
        selection_string="index==index", cdf=gaussian,
    )  # fmt: skip
    df = pd.DataFrame(
        {"cuspEmax_cal": np.linspace(1000, 2000, 50), "dt_eff": np.ones(50)}
    )
    for plot in (
        lq.plot_spectra,
        lq.plot_sf_vs_energy,
        lq.plot_classifier,
        lq.plot_drift_time_correction,
        lq.plot_lq_cut_fit,
        lq.plot_survival_fraction_curves,
    ):
        assert plot(lqcal, df) is not None
        assert isinstance(plot.data_func(lqcal, df), dict)
