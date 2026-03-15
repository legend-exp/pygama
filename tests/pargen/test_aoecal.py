from __future__ import annotations

import numpy as np
from lgdo import lh5

import pygama.pargen.AoE_cal as Coe


def test_aoe_cal(lgnd_test_data):
    # load test data here
    data = lgnd_test_data.get_path(
        "lh5/prod-ref-l200/generated/tier/dsp/cal/p03/r000/l200-p03-r000-cal-20230311T235840Z-tier_dsp.lh5"
    )

    data_df = lh5.read_as("ch1104000/dsp", data, "pd")

    data_df["AoE_Uncorr"] = data_df["A_max"] / data_df["cuspEmax"]

    data_df["cuspEmax_cal"] = data_df["cuspEmax"] * 0.155

    cal_dict = {
        "AoE_Uncorr": {
            "expression": "A_max/cuspEmax",
            "parameters": {},
        }
    }

    aoe = Coe.CalAoE(
        cal_dict,
        "cuspEmax_cal",
        lambda x: np.sqrt(1.5 + 0.1 * x),
        selection_string="index==index",
        debug_mode=True,
    )
    aoe.calibrate(data_df, "AoE_Uncorr")
    assert (
        (aoe.low_cut_val < -1.0) & (aoe.low_cut_val > -3) & (~np.isnan(aoe.low_cut_val))
    )
    assert (
        (~np.isnan(aoe.low_side_sfs.loc[2614.5]["sf"]))
        & (aoe.low_side_sfs.loc[2614.5]["sf"] < 20)
        & (aoe.low_side_sfs.loc[2614.5]["sf"] > 0)
    )
